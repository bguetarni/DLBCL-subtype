import os, math, json, argparse
import tqdm
import numpy as np
import pandas
import torch
import torch.nn as nn
import einops
from sklearn import metrics

from models.MultiModalViT import MonoModalViT
from utils import preprocess, data_loaders

def metrics_fn(args, Y_true, Y_pred):
    """
    args : script arguments
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    """

    # cross-entropy error
    error = metrics.log_loss(Y_true, Y_pred)

    # ROC AUC (per class)
    auc = dict()
    for i in range(Y_true.shape[1]):
        # select class one-hot values
        ytrue = Y_true[:,i]

        # transform probabilities from [0.5,1] to [0,1]
        # probabilities in [0,0.5] are clipped to 0
        ypred = np.clip(Y_pred[:,i], 0.5, 1) * 2 - 1
        auc_score = metrics.roc_auc_score(ytrue, ypred)
        auc.update({i: auc_score})
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

    # computes binary metrics for each class (one versus all)
    labels = pandas.read_csv(args.labels).set_index('slide')['label'].to_dict()
    for i in set(labels.values()):
        
        # statistics from sklearn confusion matrix
        tn, fp, fn, tp = multiclass_cm[i].ravel()

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
        fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        metrics_.update({
            "{}_auc".format(i): auc[i],
            "{}_precision".format(i): precision,
            "{}_recall".format(i): recall,
            "{}_fscore".format(i): fscore,
            "{}_fnr".format(i): fnr,
            })

    return metrics_

def prepare_batch(batch):
    """
    args:
        batch : batch of sequences and classes

    return:
        x : stains data processed and stacked in a list (list of torch.Tensor)
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


def make_model_predict(model, args, data):
    """
    args:
        model (torch.nn.Module): model to apply on data
        args: scrpt arguments
        data (pandas.DataFrame): test data
    """

    y_pred = []
    y_true = []
    for batch in tqdm.tqdm(np.array_split(data, indices_or_sections=math.ceil(len(data) / args.batch_size)),
                               ncols=50):
        
        # prepare batch
        x, y = prepare_batch(batch, args.dataset)
        
        # inference
        with torch.no_grad():
            slide_score = model(x[MODALITIES[0]])

        # remove attention weights
        if isinstance(slide_score, tuple):
            slide_score = slide_score[0]

        y_pred.append(slide_score.to(device='cpu').numpy())
        y_true.append(y)

    # all prediction in one array
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    return y_pred, y_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--model', type=str, required=True, help='path to the model directory')
    parser.add_argument('--output', type=str, required=True, help='path to the directory to save results')
    parser.add_argument('--labels', type=str, required=True, help='path to csv file with labels')
    parser.add_argument('--in_chans', type=int, required=True, help='number of input channels for ViT')
    parser.add_argument('--n_class', type=int, required=True, help='number of classes')
    parser.add_argument('--modalities', type=str, required=True, help='name of the modalities separated by a comma (e.g., `HES,BCL6,CD10,MUM1`) \
                        WARNING: the first modality is the one to be distilled after')
    parser.add_argument('--stain', type=str, required=True, choices=["multi", "mono"], help='multi or mono-stain training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
    parser.add_argument('--MAX_SAMPLE_PER_PATIENT', type=int, default=10000, help='number of sample per patient ot use')
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use (e.g. 0)')
    args = parser.parse_args()

    global MODALITIES
    MODALITIES = args.modalities.split(",")

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print('device: ', device)

    # get model name
    model_name = os.path.split(args.model)[1]
    print('testing model {}'.format(model_name))
    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # create model
    print('initialize model..')
    model_dict = json.load(open(os.path.join(args.model, "config.json"), 'r'))
    model = MonoModalViT(**model_dict, modality=MODALITIES[0])
    
    # load model weights
    print('load model weights..')
    model_weights = os.path.join(args.model, "ckpt.pth")
    state_dict = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # multi GPU
    model = nn.DataParallel(model).eval().to(device=device)

    # load testing data
    print('loading data..')
    data = data_loaders.DataLoader(args)    

    # model inference
    print('predicting on all data..')
    y_pred, y_true = make_model_predict(model, args, data)

    # calculates metrics
    metrics = metrics_fn(args, y_true, y_pred)

    # build model DataFrame with results
    df = pandas.DataFrame({model_name: metrics.values()}, index=metrics.keys()).transpose()

    # round floats to 2 decimals
    df = df.round(decimals=3)

    # save results in a CSV file
    df.to_csv(os.path.join(output_dir, "metrics.csv"))
