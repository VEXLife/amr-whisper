import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import create_dataloaders
from model import DeepReceiver, compute_loss, filter_invalid_targets


class LightningDeepReceiver(pl.LightningModule):
    def __init__(self, num_bits=150, learning_rate=0.001):
        super(LightningDeepReceiver, self).__init__()
        self.model = DeepReceiver(num_bits=num_bits)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        iq_wave, bin_seq, symb_seq, symb_type, symb_wid = batch  # 正确解包
        iq_wave = iq_wave.permute(0, 2, 1)  # Convert to [B, C, T] for Conv1D
        bin_seq = bin_seq.to(self.device)  # 确保标签在正确的设备上
        outputs = self(iq_wave)

        loss = compute_loss(outputs, bin_seq, ignore_index=2)  # 使用 bin_seq 作为标签，并设置 ignore_index=2
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        iq_wave, bin_seq, symb_seq, symb_type, symb_wid = batch  # 正确解包
        iq_wave = iq_wave.permute(0, 2, 1)
        bin_seq = bin_seq.to(self.device)
        outputs = self(iq_wave)

        loss = compute_loss(outputs, bin_seq, ignore_index=2)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Accuracy calculation (optional)
        filtered_outputs, filtered_targets = filter_invalid_targets(bin_seq, outputs, ignore_index=2)  # 使用 bin_seq
        total_correct = 0
        total_bits = 0
        for output, target in zip(filtered_outputs, filtered_targets):
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            total_correct += (predictions == target).sum().item()
            total_bits += target.numel()
        accuracy = total_correct / total_bits if total_bits > 0 else 0
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


# Lightning DataModule
class SignalDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, train_ratio=0.8):
        super(SignalDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        self.train_loader, self.val_loader = create_dataloaders(
            self.data_path, batch_size=self.batch_size, train_ratio=self.train_ratio
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


# Training and evaluation
if __name__ == "__main__":
    # Paths and parameters
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(SCRIPT_DIR, "train_data")
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    save_path = "lightning_deepreceiver_best.ckpt"

    # Initialize PyTorch Lightning model and datamodule
    model = LightningDeepReceiver(learning_rate=learning_rate)
    data_module = SignalDataModule(data_path, batch_size=batch_size)

    # Define a Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=20,  # 替代 progress_bar_refresh_rate
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )],
    )

    # Train the model
    print("Start training...")
    trainer.fit(model, data_module)

    # Evaluate the model
    print("Evaluating...")
    trainer.validate(model, datamodule=data_module)