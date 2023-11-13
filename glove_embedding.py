import numpy as np

def embed_glove(data, embedding_dict, pad_len, UNK):
    """Embed a dataset using GLOVE embeddings.
    data: Huggingface dataset type
    embedding_dict: dictionary containing GLOVE embedding mappings from word to vector
    pad_len: int specifying the length of each sentence (pad if necessary)
    UNK: np.ndarray specifying the embedding for words not in the embedding_dict
    """
    l = []

    # just hard-coding punctuation that should get cleaned up
    puncs = ['.', ',', '!', '.', '(', ')', '?', '"', '$']

    for sentence in data:
        for punc in puncs:
            sentence = sentence.replace(punc, '')
        vec_sentence = ''.join(letter if letter.isalnum() else ' ' for letter in sentence.lower()).split(" ")
        embed = np.transpose(embedding_dict[vec_sentence[0]] if vec_sentence[0] in embedding_dict else UNK)
        for i in vec_sentence[1:]:
            to_append = embedding_dict[i] if i in embedding_dict else UNK
            to_append = np.transpose(to_append)
            embed = np.append(embed, to_append, axis=0)
        while embed.shape[0] < pad_len:
            embed = np.append(embed, np.transpose(UNK), axis=0)
        l.append(embed[:15, :].ravel())

    return np.asarray(l)

def load_glove(path):
    """Loads GLOVE embedding dictionary from input path."""
    embedding_dict = {}

    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = np.expand_dims(vector, axis=1)
    
    return embedding_dict

def retrieve_unk(embedding_dict):
    """Computes UNK vector embedding."""
    embeddings = [embedding_dict[i] for i in embedding_dict.keys()]
    arr_embeddings = np.array(embeddings)
    return np.mean(arr_embeddings, axis=0)