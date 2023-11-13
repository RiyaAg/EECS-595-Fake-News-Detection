# Random functions - reorganize at some point.

def set_idx(example, idx):
    """Replaces all instance labels with binary labels given the intended label as input."""
    example['label'] = (example['label'] == idx)
    return example