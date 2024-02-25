import torch
import numpy as np
from torch.utils.data import Dataset


EOS_TOKEN = "|"


def preprocess_text(text_file):
    with open(text_file, "r") as f:
        text = f.readlines()
    
    text_raw = "".join(text)
    assert EOS_TOKEN not in text_raw
    
    text_with_eos_token = EOS_TOKEN + EOS_TOKEN.join(text)
    
    vocab = [EOS_TOKEN] + sorted(list(set(text_raw)))
    vocab_size = len(vocab)
            
    tokenizer = {
        "token2idx": {token: idx for idx, token in enumerate(vocab)},
        "idx2token": {idx: token for idx, token in enumerate(vocab)}
    }
    
    text_indices = [tokenizer["token2idx"][token] for token in list(text_with_eos_token)]
    full_text_array = np.array(text_indices)
    
    return vocab_size, tokenizer, full_text_array


class TextDataset(Dataset):
    def __init__(self, full_text_array, context_length, batch_size, discriminator_accumulation_steps):
        shift = np.random.randint(context_length)
        text_array = full_text_array[shift:]
        first_truncate = (text_array.shape[0] // context_length) * context_length
        truncated_array = text_array[:first_truncate].reshape(-1, context_length)
        all_padded_array = np.copy(truncated_array)
        for truncation in range(1, context_length - 1):
            padded_array = np.copy(truncated_array)
            padded_array = np.concatenate((np.zeros((padded_array.shape[0], context_length - truncation), dtype=np.int64), padded_array[:, :truncation]), axis=1)
            all_padded_array = np.concatenate((all_padded_array, padded_array), axis=0)
        second_truncate_value = context_length * batch_size * discriminator_accumulation_steps
        second_truncate = (all_padded_array.shape[0] // second_truncate_value) * second_truncate_value
        all_padded_array_truncated = all_padded_array[:second_truncate]
        self.all_padded_tensor_truncated = torch.tensor(all_padded_array_truncated, dtype=torch.int64)
    
    def __len__(self):
        return len(self.all_padded_tensor_truncated)

    def __getitem__(self, idx):
        return self.all_padded_tensor_truncated[idx, :]
        