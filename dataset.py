import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.utils.rnn as rnn_utils
import lightning as L


def recover_symb_seq_from_bin_seq(bin_seq: list[torch.LongTensor], symb_bits: int) -> torch.LongTensor:
    if symb_bits == 1:
        return torch.Tensor(bin_seq).unsqueeze(0)  # 1 bit per symbol, no need to convert
    
    if len(bin_seq) == 0:
        return torch.Tensor(0)  # Empty sequence, no need to convert
    
    # Pad the binary sequence to the nearest multiple of symb_bits
    pad_len = symb_bits - len(bin_seq) % symb_bits
    bin_seq = bin_seq + [torch.LongTensor([0])] * pad_len

    # Convert to symbol sequence
    multipliers = 2 ** torch.arange(symb_bits - 1, -1, -1).unsqueeze(0)
    bin_seq_mat = torch.stack(bin_seq).reshape(-1, symb_bits)
    symb_seq = torch.matmul(bin_seq_mat, multipliers.T).T # A two-dimensional tensor
    return symb_seq # Shape: (1, symb_seq_len)


def convert_to_bin_seq_and_pad(symb_seq, symb_bits):
    if symb_bits == 1:
        return symb_seq  # 1 bit per symbol does not need to be converted
    bin_seq = []
    for symb in symb_seq:
        bin_seq.extend([int(bit) for bit in bin(symb)[2:].zfill(symb_bits)])
    return bin_seq


def get_modulation_symb_bits(symb_type):
    """
    Get the number of bits per symbol for a given modulation type index.
    1: BPSK, 2: QPSK, 3: 8PSK, 4: MSK, 5: 8QAM, 6: 16-QAM, 7: 32-QAM, 8: 8-APSK, 9: 16-APSK, 10: 32-APSK
    """
    match symb_type:
        case 1 | 4:
            return 1
        case 2:
            return 2
        case 3 | 5 | 8:
            return 3
        case 6 | 9:
            return 4
        case 7 | 10:
            return 5
        case _:
            raise ValueError(f"Unknown modulation type index: {symb_type}")


class SignalDataset(Dataset):
    def __init__(self, data_path):
        super(SignalDataset, self).__init__()
        # Recursively find all csv files in the data_path
        self.file_list = glob.glob(os.path.join(
            data_path, '**/*.csv'), recursive=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = pd.read_csv(self.file_list[index], header=None, names=[
                           'I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])

        iq_wave = data[['I', 'Q']].values
        symb_seq = data['Code Sequence'].dropna().astype(int).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]
        bin_seq = convert_to_bin_seq_and_pad(
            symb_seq, get_modulation_symb_bits(symb_type))

        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        bin_seq = torch.tensor(bin_seq, dtype=torch.long)
        symb_seq = torch.tensor(symb_seq, dtype=torch.long)
        symb_type = torch.tensor(symb_type, dtype=torch.long)
        symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
        return iq_wave, bin_seq, symb_seq, symb_type, symb_wid


def _collate_fn(train_data):
    iq_wave, bin_seq, symb_seq, symb_type, symb_wid = zip(*train_data)
    iq_wave = rnn_utils.pad_sequence(
        iq_wave, batch_first=True, padding_value=0)
    bin_seq = rnn_utils.pad_sequence(
        bin_seq, batch_first=True, padding_value=2)
    symb_seq = rnn_utils.pad_sequence(
        symb_seq, batch_first=True, padding_value=-1)
    symb_type = torch.tensor(symb_type, dtype=torch.long)
    symb_wid = torch.tensor(symb_wid, dtype=torch.float32)
    iq_wave = torch.permute(iq_wave, [0, 2, 1])
    return iq_wave, bin_seq, symb_seq, symb_type, symb_wid


class SignalDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size=32, train_ratio=0.8, num_workers=0, collate_fn=_collate_fn):
        super(SignalDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        dataset = SignalDataset(self.data_path)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
