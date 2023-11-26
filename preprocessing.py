# Random functions - reorganize at some point.

def set_idx(example, idx):
    """Replaces all instance labels with binary labels given the intended label as input."""
    example['label'] = (example['label'] == idx)
    return example

def binarize(example):
    """Binarize LIAR Dataset."""
    label_mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5:1}
    val = example['label']
    example['label'] = label_mapping[val]
    return example