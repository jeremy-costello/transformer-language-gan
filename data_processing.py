import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset


BOS_TOKEN = "<BOS>"


def preprocess_text(text_file):
    with open(text_file, "r") as f:
        text = f.read()

    vocab = [BOS_TOKEN] + sorted(list(set(text)))
    vocab_size = len(vocab)
    
    total_length = len(list(text))
    
    tokenizer = {
        "token2idx": {token: idx for idx, token in enumerate(vocab)},
        "idx2token": {idx: token for idx, token in enumerate(vocab)}
    }
    
    token_probs = torch.zeros(vocab_size)
    for key, value in Counter(list(text)).items():
        token_probs[tokenizer["token2idx"][key]] = value / total_length
    assert token_probs.sum().item() == 1.0
    
    text_indices = [tokenizer["token2idx"][token] for token in list(text)]
    full_text_array = np.array(text_indices)
    
    return vocab_size, tokenizer, token_probs, full_text_array


class TextDataset(Dataset):
    def __init__(self, full_text_array, context_length, batch_size):
        shift = np.random.randint(context_length)
        text_array = full_text_array[shift:]
        first_truncate = (text_array.shape[0] // context_length) * context_length
        truncated_array = text_array[:first_truncate].reshape(-1, context_length)
        all_padded_array = np.copy(truncated_array)
        for truncation in range(1, context_length - 1):
            padded_array = np.copy(truncated_array)
            padded_array[:, truncation:] = 0
            all_padded_array = np.concatenate((all_padded_array, padded_array), axis=0)
        second_truncate = (all_padded_array.shape[0] // (context_length * batch_size)) * context_length * batch_size
        all_padded_array_truncated = all_padded_array[:second_truncate]
        self.all_padded_tensor_truncated = torch.tensor(all_padded_array_truncated, dtype=torch.int64)
    
    def __len__(self):
        return len(self.all_padded_tensor_truncated)

    def __getitem__(self, idx):
        return self.all_padded_tensor_truncated[idx, :]
        