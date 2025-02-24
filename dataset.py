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
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = pd.read_csv(self.file_list[index], header=None, names=[
                           'I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])

        iq_wave = data[['I', 'Q']].values
        symb_seq = data['Code Sequence'].dropna().astype(int).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]

        iq_wave, iq_wave_len = self.feature_extractor(iq_wave)
        target = self.tokenizer.encode(symb_type, symb_wid, symb_seq)

        return iq_wave, iq_wave_len, target


def collator_fn(batch):
    input_features = rnn_utils([item[0] for item in batch], batch_first=True)
    input_lengths = torch.concat([item[1] for item in batch])
    labels = rnn_utils([item[2] for item in batch],
                       batch_first=True, padding_value=-100)
    return {
        "input_features": input_features,
        "input_lengths": input_lengths,
        "labels": labels,
    }


class CustomSignalDataset(SignalDataset):
    def __init__(self, testpath, feature_extractor, tokenizer):
        # 初始化 file_list
        self.file_list = [os.path.join(testpath, file) for file in os.listdir(testpath) if file.endswith('.csv')]
        self.feature_extractor = feature_extractor  # 初始化 feature_extractor
        self.tokenizer = tokenizer  # 保存 tokenizer
        print(f"Dataset contains {len(self.file_list)} files.")  # 打印文件数量

    def __len__(self):
        return len(self.file_list)  # 返回文件列表的长度

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = pd.read_csv(file_path, header=None, names=['I', 'Q'])  # 仅加载 I 和 Q 列
        iq_wave = data[['I', 'Q']].values
        iq_wave = self.feature_extractor(iq_wave)  # 提取特征
        return iq_wave, None  # 返回 iq_wave 和 None 作为标签