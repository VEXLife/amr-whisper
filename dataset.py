import os 
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.utils.rnn as rnn_utils 

def _convert_to_bin_seq_and_pad(symb_seq, symb_bits):
    bin_seq = []
    for symb in symb_seq:
        try:
            bin_seq.extend([int(bit) for bit in bin(symb)[2:].zfill(symb_bits)])
        except:
            print(symb_seq)
            print(symb)
            raise ValueError("Invalid symbol sequence")
    return bin_seq

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
        symb_seq = data['Code Sequence'].dropna().astype(int).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]

        # Convert symbol sequence to binary sequence
        # 1：BPSK，2：QPSK，3：8PSK，4：MSK，5：8QAM，6：16-QAM，7：32-QAM，8：8-APSK，9：16-APSK，10：32-APSK
        match symb_type:
            case 1 | 4:
                bin_seq = symb_seq # BPSK and MSK do not need to be converted to binary sequence
            case 2:
                bin_seq = _convert_to_bin_seq_and_pad(symb_seq, 2)
            case 3 | 5 | 8:
                bin_seq = _convert_to_bin_seq_and_pad(symb_seq, 3)
            case 6 | 9:
                bin_seq = _convert_to_bin_seq_and_pad(symb_seq, 4)
            case 7 | 10:
                bin_seq = _convert_to_bin_seq_and_pad(symb_seq, 5)
            case _:
                raise ValueError(f"Unknown modulation type index: {symb_type}")
        
        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        bin_seq = torch.tensor(bin_seq, dtype=torch.int8)
        symb_seq = torch.tensor(symb_seq, dtype=torch.int8)
        symb_type = torch.tensor(symb_type, dtype=torch.int8)
        symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
        return iq_wave, bin_seq, symb_seq, symb_type, symb_wid

def _collate_fn(train_data):
    iq_wave, bin_seq, symb_seq, symb_type, symb_wid = zip(*train_data)
    iq_wave = rnn_utils.pad_sequence(iq_wave, batch_first=True, padding_value=0)
    bin_seq = rnn_utils.pad_sequence(bin_seq, batch_first=True, padding_value=2)
    symb_seq = rnn_utils.pad_sequence(symb_seq, batch_first=True, padding_value=-1)
    symb_type = torch.tensor(symb_type, dtype=torch.long)
    symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
    return iq_wave, bin_seq, symb_seq, symb_type, symb_wid

def create_dataloaders(data_path, batch_size=32, train_ratio=0.8):
    """
    Creates dataloaders from the signal dataset.

    Args:
    data_path (str): Path to the directory containing the dataset.
    batch_size (int, optional): Number of samples per batch. Default is 32.
    train_ratio (float, optional): Ratio of the dataset to include in the train split. Default is 0.8.

    Returns:
    train_loader (DataLoader): DataLoader for the training set.
    val_loader (DataLoader): DataLoader for the validation set.
    """
    dataset = SignalDataset(data_path)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return train_loader, val_loader
