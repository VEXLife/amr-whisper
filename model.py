import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

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

        # Output layers for binary classifiers
        self.binary_classifiers = nn.ModuleList([nn.Linear(num_feats * 2, 2) for _ in range(num_feats)])

        # Output layers for MSC classifier
        self.msc_classifier = nn.Linear(num_feats * 2, 10)

        # Output layers for symbol width regressor
        self.symbol_width_regressor = nn.Linear(num_feats * 2, 1)

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

        # Binary classification for each bit
        output_bits_logits = []
        for classifier in self.binary_classifiers:
            output_bits_logits.append(classifier(features))

        # MSC classification
        msc_logits = self.msc_classifier(features)

        # Symbol width regression
        symbol_width_logits = self.symbol_width_regressor(features)

        return output_bits_logits, msc_logits, symbol_width_logits
    
class LitDenseNet(L.LightningModule):
    def __init__(self, **kwargs):
        super(LitDenseNet, self).__init__()
        self.model = DenseNet(**kwargs)
        
    def training_step(self, batch, batch_idx):
        iq_wave_batch, bin_seq_batch, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        output_bits_logits, msc_logits, symbol_width_logits = self.model(iq_wave_batch)
        seq_loss = F.cross_entropy(output_bits_logits, bin_seq_batch)
        msc_loss = F.cross_entropy(msc_logits, symb_type_batch)
        width_loss = F.mse_loss(symbol_width_logits, symb_wid_batch)
        loss = seq_loss + msc_loss + width_loss
        self.log('train/seq_loss', seq_loss)
        self.log('train/msc_loss', msc_loss)
        self.log('train/width_loss', width_loss)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        iq_wave_batch, bin_seq_batch, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        output_bits_logits, msc_logits, symbol_width_logits = self.model(iq_wave_batch)
        seq_loss = F.cross_entropy(output_bits_logits, bin_seq_batch)
        msc_loss = F.cross_entropy(msc_logits, symb_type_batch)
        width_loss = F.mse_loss(symbol_width_logits, symb_wid_batch)
        loss = seq_loss + msc_loss + width_loss
        self.log('val/seq_loss', seq_loss)
        self.log('val/msc_loss', msc_loss)
        self.log('val/width_loss', width_loss)
        self.log('val/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)
