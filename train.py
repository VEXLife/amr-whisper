import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import lightning as L

# 自定义数据集
class SignalDataset(Dataset):
    def __init__(self, data_file):
        data = pd.read_csv(data_file, header=None, names=['I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])
        data = data.dropna()

        # 输入特征
        self.I = data['I'].values
        self.Q = data['Q'].values
        self.X = np.stack((self.I, self.Q), axis=1)

        # 目标变量
        self.code_sequences = data['Code Sequence'].apply(lambda x: [int(i) for i in str(int(x))]).tolist()
        self.modulation_types = data['Modulation Type'].values.astype(int)
        self.symbol_widths = data['Symbol Width'].values.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        code_seq = self.code_sequences[idx]
        mod_type = self.modulation_types[idx]
        symbol_width = self.symbol_widths[idx]

        return x, torch.tensor(code_seq, dtype=torch.long), mod_type, symbol_width

# 填充批处理数据
def collate_fn(batch):
    xs, code_seqs, mod_types, symbol_widths = zip(*batch)

    xs = torch.stack([torch.tensor(x, dtype=torch.float32) for x in xs])

    code_seqs_padded = pad_sequence(code_seqs, batch_first=True, padding_value=0)
    code_seq_masks = (code_seqs_padded != 0)

    mod_types = torch.tensor(mod_types, dtype=torch.long)
    symbol_widths = torch.tensor(symbol_widths, dtype=torch.float32)

    return xs, code_seqs_padded, code_seq_masks, mod_types, symbol_widths

# 创建数据集和数据加载器
dataset = SignalDataset('train_data/8QAM/data_1.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 定义模型
class SignalModel(L.LightningModule):
    def __init__(self, code_vocab_size, num_mod_types):
        super(SignalModel, self).__init__()
        self.save_hyperparameters()

        # 输入层
        self.input_fc = torch.nn.Linear(2, 128)

        # Transformer 编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 码序列解码器
        self.code_decoder = torch.nn.Linear(128, code_vocab_size)

        # 调制类型分类器
        self.mod_classifier = torch.nn.Linear(128, num_mod_types)

        # 码元宽度回归器
        self.symbol_regressor = torch.nn.Linear(128, 1)

        # 损失函数
        self.code_loss_fn = torch.nn.CrossEntropyLoss()
        self.mod_loss_fn = torch.nn.CrossEntropyLoss()
        self.symbol_loss_fn = torch.nn.MSELoss()

    def forward(self, x, code_seq, code_seq_mask):
        # x: (batch_size, feature_dim)
        x = self.input_fc(x)
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, d_model)

        # Transformer 编码
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        encoder_output = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # 取最后一个时间步
        encoder_output = encoder_output[:, -1, :]  # (batch_size, d_model)

        # 码序列预测
        code_logits = self.code_decoder(encoder_output)  # (batch_size, code_vocab_size)

        # 调制类型预测
        mod_logits = self.mod_classifier(encoder_output)  # (batch_size, num_mod_types)

        # 码元宽度预测
        symbol_pred = self.symbol_regressor(encoder_output).squeeze(1)  # (batch_size)

        return code_logits, mod_logits, symbol_pred

    def training_step(self, batch, batch_idx):
        x, code_seq, code_seq_mask, mod_types, symbol_widths = batch

        code_logits, mod_logits, symbol_pred = self(x, code_seq, code_seq_mask)

        # 计算损失
        code_loss = self.code_loss_fn(code_logits, code_seq[:, 0])
        mod_loss = self.mod_loss_fn(mod_logits, mod_types)
        symbol_loss = self.symbol_loss_fn(symbol_pred, symbol_widths)

        loss = code_loss + mod_loss + symbol_loss

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 获取码序列词汇大小和调制类型数量
code_vocab_size = max([max(seq) for seq in dataset.code_sequences]) + 1
num_mod_types = dataset.modulation_types.max() + 1

# 初始化模型
model = SignalModel(code_vocab_size=code_vocab_size, num_mod_types=num_mod_types)

# 训练模型
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, dataloader)
