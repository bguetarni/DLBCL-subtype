import os, pickle, time, math, argparse, json
import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn

from models.MultiModalViT import MultiModalViT, MonoModalViT
from utils import utils, preprocess, data_loaders

def StudentLoss(y, pred_student, pred_teacher):
    """
    Loss function for knowledge distillation (handle soft and hard distillation)
    
    args:
        y (torch.Tensor): ground-truth
        pred_student (torch.Tensor): student output
        pred_teacher (torch.Tensor): teacher output
    """
    
    # classification loss with true labels
    classification_loss = utils.smooth_cross_entropy(pred_student, y, args.smooth_label)

    # logit distillation loss
    if TRAIN_PARAMS['DISTIL_HARD']:
        hard_labels = pred_teacher.argmax(dim=-1)
        distil_loss = utils.smooth_cross_entropy(pred_student, hard_labels, args.smooth_label)
    else:
        softened_softmax = lambda x : nn.functional.softmax(x / TRAIN_PARAMS['DISTIL_SOFTMAX_TEMP'], dim=-1)
        distil_loss = nn.functional.kl_div(torch.log(softened_softmax(pred_student)), softened_softmax(pred_teacher), reduction='batchmean')
        distil_loss = (TRAIN_PARAMS['DISTIL_SOFTMAX_TEMP'] ** 2) * distil_loss

    loss = TRAIN_PARAMS['DISTIL_TEACHER_WEIGHT'] * distil_loss + TRAIN_PARAMS['DISTIL_TRUE_WEIGHT'] * classification_loss
    return loss

def prepare_batch(batch):
    """
    Prepare batch for training.
    
    Args:
        batch : batch of sequences and classes

    Return:
        x : stains data processed and stacked in a list (list of dict (stain) of torch.Tensor)
        y : ground-truth
    """

    x = batch["x"].values
    y = batch["y"].values
    
    permutation = 's h w c -> s c h w'
    transform_seq = lambda seq : einops.rearrange(preprocess.normalize(torch.tensor(seq, dtype=torch.float)), permutation)

    # convert to tensor
    # normalize inputs
    # permute channel position
    x = map(lambda d : {k: transform_seq(v) for k, v in d.items()}, x)
    x = list(x)
    x = {s: torch.stack([i[s] for i in x], dim=0) for s in MODALITIES}
        
    y = torch.tensor(y).to(dtype=torch.int64)

    # x is list of dict :
    # [ {stain: np.ndarray(s,psize,psize,3), stain: np.ndarray(s,psize,psize,3), ...},
    #   {stain: np.ndarray(s,psize,psize,3), stain: np.ndarray(s,psize,psize,3), ...},
    #    ...]
    return x, y

def validation(args, model, validation_data):
    """
    Validation stage
    
    Args:
        args : arguments
        model : model to train
        validation_data : iterable that contains validation data that is already splitted in batches

    Return:
        loss : total loss
    """
    
    Y_pred_slide, Y_slide = [], []
    for batch in tqdm.tqdm(np.array_split(validation_data, indices_or_sections=math.ceil(len(validation_data) / args.batch_size)),
                           ncols=50):
        
        # prepare data
        x, y_slide = prepare_batch(batch)
        
        # send input to device
        x = {k: v.to(device=device) for k, v in x.items()}
        
        with torch.no_grad():
            if args.stain == "mono":
                x = x[MODALITIES[0]]
            
            slide_score = model(x)

        if isinstance(slide_score, tuple):
            slide_score = slide_score[0]

        Y_pred_slide.append(slide_score.to('cpu'))
        Y_slide.append(y_slide)

    # concatenate batches predictions and labels
    Y_pred_slide = torch.cat(Y_pred_slide)
    Y_slide = torch.cat(Y_slide)
    
    # compute loss
    loss = nn.functional.cross_entropy(Y_pred_slide, Y_slide)
    
    return loss.to('cpu').item()

def train_step(args, batch, model, epoch, n_iter, teacher=None):
    """
    Train step for a single batch
    
    Args:
        args : arguments
        batch : a single batch
        model : model to train
        epoch : current epoch
        n_iter : current gradient accumulation step
        teacher : teacher model if training student

    Return:
        batch_loss : loss on batch
    """

    # prepare batch (data augmentation, convert hot-index to vector, normalize...)
    x, y = prepare_batch(batch)
    y = y.to(device)

    # input to device
    x = {k: v.to(device=device) for k, v in x.items()}

    # forward
    if teacher is None:
        if args.stain == 'mono':
            x = x[MODALITIES[0]]
        
        slide_score = model(x)
        
        if isinstance(slide_score, tuple):
            slide_score = slide_score[0]

        # computes batch loss
        batch_loss = utils.smooth_cross_entropy(slide_score, y, args.smooth_label)
    else:
        # student forward
        student_x = x[MODALITIES[0]]
        slide_score = model(student_x)
        
        if isinstance(slide_score, tuple):
            slide_score = slide_score[0]

        # teacher forward
        with torch.no_grad():
            teacher_slide_score = teacher(x)

            if isinstance(teacher_slide_score, tuple):
                teacher_slide_score = teacher_slide_score[0]
    
        # student loss
        if epoch < TRAIN_PARAMS['DISTIL_N_EPOCHS']:
            batch_loss = StudentLoss(y, slide_score, teacher_slide_score)
        else:
            batch_loss = utils.smooth_cross_entropy(slide_score, y, args.smooth_label)

    # backpropagation
    batch_loss /= args.n_accumulation_step
    batch_loss.backward()

    # update step
    if (n_iter + 1) % args.n_accumulation_step == 0:

        # clip gradients
        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # optimizer update and reset gradients
        optimizer.step()
        optimizer.zero_grad()

    return batch_loss.to('cpu').item()

if __name__ == '__main__':
    # general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_chans', type=int, required=True, help='number of input channels for ViT')
    parser.add_argument('--n_class', type=int, required=True, help='number of classes')
    parser.add_argument('--modalities', type=str, required=True, help='name of the modalities separated by a comma (e.g., `HES,BCL6,CD10,MUM1`) \
                        WARNING: the first modality is the one to be distilled after')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save model weights and metadata')
    parser.add_argument('--labels', type=str, required=True, help='path to csv file with labels')
    parser.add_argument('--stain', type=str, required=True, choices=["multi", "mono"], help='multi or mono-stain training')
    parser.add_argument('--teacher', type=str, default=None, help='path to the teacher if training the student')
    parser.add_argument('--name', type=str, required=True, help='name of the model')
    parser.add_argument('--validation_factor', type=float, default=0.2, help='factor to isolate valdiation data from train data')
    parser.add_argument('--description', type=str, default='', help='description of the model and train strategy')
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use (e.g. 0,1)')
    
    # training arguments
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--n_accumulation_step', type=int, default=1, help='steps to accumulate gradient')
    parser.add_argument('--clip_grad', type=float, default=3.0, help='gradient clipping norm')
    parser.add_argument('--smooth_label', type=float, default=0.1, help='label smoothing value')
    parser.add_argument('--class_balance', action="store_true", help='apply class balance')
    parser.add_argument('--lambda_', type=float, default=0.0, help='lambda for HES/IHC aggregation')
    args = parser.parse_args()

    if args.stain == "multi":
        print("training mul-stain teacher model: {}".format(args.name))
    elif not args.teacher is None:
        print("training mono-stain student model: {}".format(args.name))
    else:
        print("training mono-stain model without teacher: {}".format(args.name))

    global MODALITIES
    MODALITIES = args.modalities.split(",")

    global TRAIN_PARAMS
    TRAIN_PARAMS = dict(
        
        # multi-modal model configuration (parameters, etc.)
        teacher_config = dict(
            vit_config = dict(in_chans=args.in_chans, embed_dim=128, depth=3, num_heads=2, mlp_ratio=4),
            num_classes = args.num_classes,
        ),

        # mono-modal model configuration (parameters, etc.)
        student_config = dict(
            vit_config = dict(in_chans=args.in_chans, embed_dim=128, depth=3, num_heads=2, mlp_ratio=3),
            num_classes = args.num_classes,
        ),

        # architecture to use
        teacher_arch = MultiModalViT,
        student_arch = MonoModalViT,

        # distillation parameters
        DISTIL_HARD = True,
        DISTIL_N_EPOCHS = math.inf,   # early stopped knowledge distillation (https://doi.org/10.48550/arXiv.1910.01348)
        DISTIL_SOFTMAX_TEMP = 3.0,
        DISTIL_TRUE_WEIGHT = 0.5,
        DISTIL_TEACHER_WEIGHT = 0.5,

        # training hyperparameters
        optimizer_class = 'torch.optim.Adam',
        teacher_optimizer_opts = dict(lr=8e-5),
        student_optimizer_opts = dict(lr=1e-5),
        scheduler_class = 'torch.optim.lr_scheduler.StepLR',
        scheduler_opts = dict(step_size=5, gamma=0.5),
    )
    TRAIN_PARAMS['teacher_config'].update({'lambda': args.lambda_})

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print('device: ', device)
    
    # create model folder
    model_path = os.path.join(args.output, args.name)
    os.makedirs(model_path, exist_ok=True)

    # save model config to json
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        if args.stain == "multi":
            json.dump(TRAIN_PARAMS['teacher_config'], f)
        else:
            json.dump(TRAIN_PARAMS['student_config'], f)

    # build model and load base parameters
    if args.stain == "multi":
        model = TRAIN_PARAMS['teacher_arch'](**TRAIN_PARAMS['teacher_config'], modalities=MODALITIES)
    else:
        model = TRAIN_PARAMS['student_arch'](**TRAIN_PARAMS['student_config'], modality=MODALITIES[0])
        
        if not args.teacher is None:
            # load teacher configuration
            with open(os.path.join(args.teacher, "config.json"), 'r') as f:
                teacher_dict = json.load(f)
            
            # load teacher model
            teacher = TRAIN_PARAMS['teacher_arch'](**teacher_dict, modalities=MODALITIES)
            state_dict = torch.load(os.path.join(args.teacher, "ckpt.pth"), map_location=torch.device('cpu'))
            teacher.load_state_dict(state_dict, strict=False)
    
    # multi GPU
    model = nn.DataParallel(model).to(device=device)
    if not args.teacher is None:
        teacher = nn.DataParallel(teacher).eval().to(device=device)

    # optimizer and learning rate scheduler
    if args.stain == "multi":
        optimizer = eval(TRAIN_PARAMS['optimizer_class'])(model.parameters(), **TRAIN_PARAMS['teacher_optimizer_opts'])
        scheduler = eval(TRAIN_PARAMS['scheduler_class'])(optimizer, **TRAIN_PARAMS['scheduler_opts'])
    else:
        optimizer = eval(TRAIN_PARAMS['optimizer_class'])(model.parameters(), **TRAIN_PARAMS['student_optimizer_opts'])

    with open(os.path.join(model_path, 'log'), 'w') as logger:

        # log the date and time
        logger.write(time.strftime("%x %X"))

        # log the training parameters
        logger.writelines(['\n{} : {}'.format(k,v) for k, v in TRAIN_PARAMS.items()])

        # log the training parameters
        logger.writelines(['\n{} : {}'.format(k,v) for k, v in vars(args).items()])

        # log the model size
        msg = ['\n\nmodel paramaters detail :\n']
        for n, submodule in model.module.named_children():
            msg.append('------ {}: {}\n'.format(n, utils.model_parameters_count(submodule)))

        msg.append('total : {}\n'.format(utils.model_parameters_count(model)))
        logger.writelines(msg)
    
    ###################   DATA LOADING   ##################
    train_data, test_data = data_loaders.DataLoader(args, validation_factor=args.validation_factor)
        
    print('\n train data :', len(train_data))
    with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log train data total
            logger.write('\ntrain data : {}'.format(len(train_data)))

    print('\n validation data :', len(test_data))
    with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log train data total
            logger.write('\nvalidation data : {}'.format(len(test_data)))
    ########################################################

    n_iter = 0
    train_loss = []
    val_loss = []
    for epoch in range(args.n_epoch):
        print('\nepoch ', epoch+1)

        print('train')
        model.train()
        loss = []
        for batch in tqdm.tqdm(np.array_split(train_data, indices_or_sections=math.ceil(len(train_data) / args.batch_size)),
                               ncols=50):

            # train step
            if args.teacher is None:
                batch_loss = train_step(args, batch, model, epoch, n_iter)
            else:
                batch_loss = train_step(args, batch, model, epoch, n_iter, teacher)
            loss.append(batch_loss)
            n_iter += 1
            
        train_loss.append(np.mean(loss))
        
        print('validation')
        model.eval()
        loss = validation(args, model, test_data)
        
        # learning rate schedule
        if args.stain == "multi":
            scheduler.step()
        
        # append validation metrics
        val_loss.append(loss)
        
        # save the weights of the model if current validation error is lowest
        if val_loss[-1] == min(val_loss):
            torch.save(model.module.state_dict(), os.path.join(model_path, 'ckpt.pth'))
        
        # save results
        results = {'train loss' : train_loss, 'validation loss': val_loss}
        with open(os.path.join(model_path, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log the date and time of end of epoch
            logger.write('\nend of epoch {} : {}'.format(epoch+1, time.strftime("%x %X")))
