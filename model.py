import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataset import get_modulation_symb_bits

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], dim=1)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            BasicBlock(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_feats=150):
        super(DenseNet, self).__init__()
        self.initial_conv = nn.Conv1d(2, 64, kernel_size=5, padding=2)  # Input has 2 channels (Re, Im)

        # Define DenseNet structure
        self.transition1 = TransitionBlock(64, 128)
        self.dense1 = DenseBlock(2, 128, 128)

        self.transition2 = TransitionBlock(128 + 2 * 128, 64)
        self.dense2 = DenseBlock(3, 64, 64)

        self.transition3 = TransitionBlock(64 + 3 * 64, 64)
        self.dense3 = DenseBlock(4, 64, 64)

        self.transition4 = TransitionBlock(64 + 4 * 64, 64)
        self.dense4 = DenseBlock(3, 64, 64)

        self.final_conv = nn.Conv1d(64 + 3 * 64, num_feats, kernel_size=5, padding=2)

        # Global Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # DenseNet layers
        x = self.transition1(x)
        x = self.dense1(x)

        x = self.transition2(x)
        x = self.dense2(x)

        x = self.transition3(x)
        x = self.dense3(x)

        x = self.transition4(x)
        x = self.dense4(x)

        # Final convolution
        x = self.final_conv(x)

        # Global pooling
        max_pooled = self.global_max_pool(x).squeeze(-1)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)

        # Concatenate pooling results
        features = torch.cat([max_pooled, avg_pooled], dim=1)

        return features
    
class LitDenseNet(L.LightningModule):
    def __init__(self, num_feats=150):
        super(LitDenseNet, self).__init__()
        self.encoder = DenseNet(num_feats)

        # Output layers for binary classifiers
        self.binary_classifiers = nn.ModuleList([nn.Linear(num_feats * 2, 3) for _ in range(num_feats)])

        # Output layers for symb_type classifier
        self.symb_type_classifier = nn.Linear(num_feats * 2, 10)

        # Output layers for symbol width regressor
        self.symbol_width_regressor = nn.Linear(num_feats * 2, 1)
        
    def training_step(self, batch, batch_idx):
        iq_wave_batch, bin_seq_batch, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        features = self.encoder(iq_wave_batch)

        # Binary classification for each bit
        output_bits_logits = []
        for classifier in self.binary_classifiers:
            output_bits_logits.append(classifier(features))

        # symb_type classification
        symb_type_logits = self.symb_type_classifier(features)

        # Symbol width regression
        symbol_width_logits = self.symbol_width_regressor(features)
        
        seq_loss = F.cross_entropy(output_bits_logits, bin_seq_batch)
        symb_type_loss = F.cross_entropy(symb_type_logits, symb_type_batch)
        width_loss = F.mse_loss(symbol_width_logits, symb_wid_batch)
        loss = seq_loss + symb_type_loss + width_loss
        self.log('train/seq_loss', seq_loss)
        self.log('train/type_loss', symb_type_loss)
        self.log('train/width_loss', width_loss)
        self.log('train/loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        iq_wave_batch, bin_seq_batch, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        features = self.encoder(iq_wave_batch)

        # Binary classification for each bit
        output_bits = []
        for classifier in self.binary_classifiers:
            bit = torch.argmax(classifier(features), dim=1)
            if bit == 2:
                break
            output_bits.append(bit)

        # Symbol modulation type classification
        symb_type = torch.argmax(self.symb_type_classifier(features), dim=1)

        # Symbol width regression
        symbol_width = self.symbol_width_regressor(features)
        return output_bits, symb_type, symbol_width

    def validation_step(self, batch, batch_idx):
        _, _, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        output_bits_hat, symb_type_hat, symbol_width_hat = self.predict_step(batch, batch_idx)
        batch_size = symb_seq_batch.size(0)

        mt_score = (symb_type_hat == symb_type_batch).float().mean() * 100
        self.log('val/mt_score', mt_score)

        er = torch.abs((symb_wid_batch - symbol_width_hat) / symb_wid_batch)
        sw_score = torch.clip(100 - (er - 0.05) / 0.15 * 100, 0, 100).mean()
        self.log('val/sw_score', sw_score)

        cq = 0
        for i in range(batch_size):
            symb_seq = _recover_symb_seq_from_bin_seq(output_bits_hat[i], get_modulation_symb_bits(symb_type_batch[i]))
            cs = torch.cosine_similarity(torch.tensor(symb_seq), symb_seq_batch[i])
            cq += torch.clip((cs - 0.7) / 0.25 * 100, 0, 100) / batch_size
        self.log('val/cq_score', cq)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def _recover_symb_seq_from_bin_seq(bin_seq, symb_bits):
    if symb_bits == 1:
        return bin_seq # 1 bit per symbol, no need to convert
    symb_seq = []
    for i in range(0, len(bin_seq), symb_bits):
        symb_seq.append(int(''.join(map(str, bin_seq[i:i + symb_bits])), 2))
    return symb_seq