import os 
import glob
import numpy as np
import pandas as pd
import itertools
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils 

class SignalDataset(Dataset):
    def __init__(self, data_path):
        super(SignalDataset, self).__init__()
        # Recursively find all csv files in the data_path
        self.file_list = glob.glob(os.path.join(data_path, '**/*.csv'), recursive=True)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data = pd.read_csv(self.file_list[index], header=None, names=['I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])
        
        iq_wave = data[['I', 'Q']].values
        symb_seq = data['Code Sequence'].fillna(-1).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]
        
        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        symb_seq = torch.tensor(symb_seq, dtype=torch.long)
        symb_type = torch.tensor(symb_type, dtype=torch.long)
        symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
        return iq_wave, symb_seq, symb_type, symb_wid

def collate_fn(train_data):
    iq_wave, symb_seq, symb_type, symb_wid = zip(*train_data)
    iq_wave = rnn_utils.pad_sequence(iq_wave, batch_first=True, padding_value=0)
    symb_seq = rnn_utils.pad_sequence(symb_seq, batch_first=True, padding_value=-1)
    symb_mask = (symb_seq != -1)
    symb_type = torch.tensor(symb_type, dtype=torch.long)
    symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
    return iq_wave, symb_seq, symb_mask, symb_type, symb_wid

# Create dataset and dataloader
def create_dataloader(data_path, batch_size=32, train_ratio=0.8):
    dataset = SignalDataset(data_path)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
