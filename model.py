import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from dataset import get_modulation_symb_bits, recover_symb_seq_from_bin_seq


class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, growth_rate,
                              kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], dim=1)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=5, padding=2)
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
        # Input has 2 channels (Re, Im)
        self.initial_conv = nn.Conv1d(2, 64, kernel_size=5, padding=2)

        # Define DenseNet structure
        self.transition1 = TransitionBlock(64, 128)
        self.dense1 = DenseBlock(2, 128, 128)

        self.transition2 = TransitionBlock(128 + 2 * 128, 64)
        self.dense2 = DenseBlock(3, 64, 64)

        self.transition3 = TransitionBlock(64 + 3 * 64, 64)
        self.dense3 = DenseBlock(4, 64, 64)

        self.transition4 = TransitionBlock(64 + 4 * 64, 64)
        self.dense4 = DenseBlock(3, 64, 64)

        self.final_conv = nn.Conv1d(
            64 + 3 * 64, num_feats, kernel_size=5, padding=2)

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
    def __init__(self, num_feats=150, lr=1e-3, max_bits=32):
        super(LitDenseNet, self).__init__()
        self.encoder = DenseNet(num_feats)

        # Output layers for binary classifier
        self.binary_classifier = nn.Linear(num_feats * 2, max_bits * 2)

        # Output layers for symb_type classifier
        self.symb_type_classifier = nn.Linear(num_feats * 2, 10)

        # Output layers for symbol width regressor
        self.symbol_width_regressor = nn.Linear(num_feats * 2, 1)

        self.lr = lr
        self.max_bits = max_bits

        self.save_hyperparameters("num_feats", "lr", "max_bits")

    def training_step(self, batch, batch_idx):
        iq_wave_batch, bin_seq_batch, _, symb_type_batch, symb_wid_batch = batch
        features = self.encoder(iq_wave_batch)

        # Binary classification for each bit
        output_bits_logits = self.binary_classifier(features)
        output_bits_logits = rearrange(output_bits_logits, 'b (cls s) -> b cls s', cls=2)

        # symb_type classification
        symb_type_logits = self.symb_type_classifier(features)

        # Symbol width regression
        symbol_width_logits = self.symbol_width_regressor(features)

        # Pad binary sequence to match output size
        assert bin_seq_batch.size(1) <= self.max_bits, f"Binary sequence length {bin_seq_batch.size(1)} is greater than max_bits {self.max_bits}"
        bin_seq_batch_padded = F.pad(
            bin_seq_batch, (0, self.max_bits - bin_seq_batch.size(1)), "constant", 2)
        seq_loss = F.cross_entropy(output_bits_logits, bin_seq_batch_padded, ignore_index=2)
        symb_type_loss = F.cross_entropy(symb_type_logits, symb_type_batch - 1)
        width_loss = F.mse_loss(symbol_width_logits,
                                symb_wid_batch.unsqueeze(1))
        loss = seq_loss + symb_type_loss + width_loss
        self.log('train/seq_loss', seq_loss)
        self.log('train/type_loss', symb_type_loss)
        self.log('train/width_loss', width_loss)
        self.log('train/loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        iq_wave_batch, _, _, _, _ = batch
        features = self.encoder(iq_wave_batch)

        # Binary classification for each bit
        output_bits_logits = self.binary_classifier(features)        
        output_bits_logits = rearrange(output_bits_logits, 'b (cls s) -> b cls s', cls=2)
        output_bits_hat = torch.argmax(output_bits_logits, dim=1)

        # Symbol modulation type classification
        symb_type = torch.argmax(self.symb_type_classifier(features), dim=1)

        # Symbol width regression
        symbol_width = self.symbol_width_regressor(features)
        return output_bits_hat, symb_type, symbol_width

    def validation_step(self, batch, batch_idx):
        iq_wave_batch, _, symb_seq_batch, symb_type_batch, symb_wid_batch = batch
        features = self.encoder(iq_wave_batch)

        # Binary classification for each bit
        output_bits_hat, symb_type_hat, symbol_width_hat = self.predict_step(
            iq_wave_batch, batch_idx
        )
        batch_size = symb_seq_batch.size(0)

        mt_score = (symb_type_hat == symb_type_batch - 1).float().mean() * 100
        self.log('val/mt_score', mt_score)

        er = torch.abs((symb_wid_batch - symbol_width_hat.T) / symb_wid_batch)
        sw_score = torch.clamp(100 - (er - 0.05) / 0.15 * 100, 0, 100).mean()
        self.log('val/sw_score', sw_score)

        cq_score = 0
        device = features.device  # Get the device from features
        for i in range(batch_size):
            # Ensure symb_type_batch[i] is on CPU to get the item, then use the device for tensors
            symb_type_val = symb_type_batch[i].item()
            symb_seq_hat = recover_symb_seq_from_bin_seq(
                output_bits_hat[i], 
                get_modulation_symb_bits(symb_type_val), 
                device
            )

            # Remove -1 elements from the ground truth
            symb_seq_ground_truth = symb_seq_batch[i][symb_seq_batch[i] != -1]
            symb_seq_ground_truth_len = symb_seq_ground_truth.numel()
            symb_seq_hat_len = symb_seq_hat.numel()

            if symb_seq_ground_truth_len == 0 or symb_seq_hat_len == 0:
                continue
            symb_seq_hat = F.pad(symb_seq_hat, (0, symb_seq_ground_truth_len - symb_seq_hat_len), "constant", 0)

            cs = torch.cosine_similarity(
                symb_seq_hat.float(), symb_seq_ground_truth.float().unsqueeze(0)
            )
            cq_score += torch.clamp((cs - 0.7) / 0.25 * 100, 0, 100) / batch_size
        self.log('val/cq_score', cq_score)

        score = 0.2 * mt_score + 0.3 * sw_score + 0.5 * cq_score
        self.log('val/score', score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
