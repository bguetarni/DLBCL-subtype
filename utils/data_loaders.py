import os, glob
import pandas

def load_slide_dir(path_, modalities):
    """
        args:
            path_ (str): path to slide directory
            modalities (List[str]): list of modalities to load
        
        returns a list like:
        [
            {'stain 1': path/to/seq, 'stain 2': path/to/seq, ...},
            {'stain 1': path/to/seq, 'stain 2': path/to/seq, ...},
            ...
        ]
    """

    slide_data = []
    for region in os.listdir(path_):

        # get list of sequences for each stain
        samples = {}
        for stain in os.listdir(os.path.join(path_, region)):
            if stain in modalities:
                samples.update({stain: glob.glob(os.path.join(path_, region, stain, "*.npy"))})
        
        # creates bags of sequences
        bags = []
        while all(samples.values()):   # check all stain have at least one sequence
            one_sample = {stain: seq.pop() for stain, seq in samples.items()}
            bags.append(one_sample)

        slide_data.extend(bags)

    return slide_data

def DataLoader(args, validation_factor=None, **kwargs):
    labels = pandas.read_csv(args.labels).set_index('slide')['label'].to_dict()
    modalities = args.modalities.split(",")
    
    # go through data folder
    data = []
    for slide_dir in os.listdir(args.data):
        samples = load_slide_dir(os.path.join(args.data, slide_dir), modalities)
        for s in samples:
            data.append({"x": s, "y": labels[slide_dir]})
    
    data = pandas.DataFrame(data)

    # class balancing
    if args.class_balance:
        class_counts = []
        for c in data["y"].unique():
            class_counts.append(len(data[data["y"] == c]))
    
        # get minimum between classes number of samples
        N = min(class_counts)

        # randomly sample N examples per class
        new_data = pandas.DataFrame([])
        for c in data["y"].unique():
            samples = data[data["y"] == c].sample(n=N)
            new_data = pandas.concat((new_data, samples))
        
        data = new_data

    # shuffle data
    data = data.sample(frac=1)

    if validation_factor:   # train
        n = int(validation_factor * len(data))
        validation_data = data[:n]
        train_data = data[n:]
        return train_data, validation_data
    else:   # test
        return data
