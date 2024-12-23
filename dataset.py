import glob
import os

import pandas as pd
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence as rnn_utils
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(SignalDataset, self).__init__()
        # Recursively find all csv files in the data_path
        self.file_list = glob.glob(os.path.join(
            data_path, '**/*.csv'), recursive=True)
        self.cache = {}  # Dictionary for caching data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        data = pd.read_csv(self.file_list[index], header=None, names=[
                           'I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])

        iq_wave = data[['I', 'Q']].values
        symb_seq = data['Code Sequence'].dropna().astype(int).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]

        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        iq_wave = rearrange(iq_wave, 't c -> c t')
        # Pad the features to 2048
        iq_wave = torch.nn.functional.pad(
            iq_wave, (0, 2048 - iq_wave.shape[1]), mode='constant', value=0)

        target = self.tokenizer.encode(symb_type, symb_wid, symb_seq)
        # Cache processed data
        self.cache[index] = (iq_wave, target)

        return iq_wave, target


def _collator_fn(batch):
    input_features = rnn_utils([item[0] for item in batch], batch_first=True)
    labels = rnn_utils([item[1] for item in batch],
                       batch_first=True, padding_value=-100)
    return {
        "input_features": input_features,
        "labels": labels,
    }
