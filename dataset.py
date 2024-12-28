import glob
import os

import pandas as pd
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence as rnn_utils
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, data_path, feature_extractor, tokenizer):
        super(SignalDataset, self).__init__()
        # Recursively find all csv files in the data_path
        self.file_list = glob.glob(os.path.join(
            data_path, '**/*.csv'), recursive=True)
        self.cache = {}  # Dictionary for caching data
        self.feature_extractor = feature_extractor
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

        iq_wave = self.feature_extractor(iq_wave)
        target = self.tokenizer.encode(symb_type, symb_wid, symb_seq)
        # Cache processed data
        self.cache[index] = (iq_wave, target)

        return iq_wave, target


def collator_fn(batch):
    input_features = rnn_utils([item[0] for item in batch], batch_first=True)
    labels = rnn_utils([item[1] for item in batch],
                       batch_first=True, padding_value=-100)
    return {
        "input_features": input_features,
        "labels": labels,
    }


class CustomSignalDataset(SignalDataset): #用于infer的时候读取文件
    """
    自定义 SignalDataset，忽略不存在的列，只处理 I 和 Q。
    """
    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = pd.read_csv(file_path, header=None, names=['I', 'Q'])  # 仅加载 I 和 Q 列
        iq_wave = data[['I', 'Q']].values
        iq_wave = self.feature_extractor(iq_wave)  # 提取特征
        return iq_wave, None  # 返回 iq_wave，忽略目标值