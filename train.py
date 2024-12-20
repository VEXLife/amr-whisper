import fire
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import create_dataloaders
from model import DeepReceiver, compute_loss, filter_invalid_targets


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
def train(data_path, batch_size=32, num_epochs=20, learning_rate=0.001, save_path="lightning_deepreceiver_best.ckpt"):
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

if __name__ == "__main__":
    fire.Fire(train)