import fire
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import SignalDataModule
from model import LitDenseNet

def train(*, data_path, 
          batch_size=32, 
          num_epochs=20, 
          num_features=150, 
          learning_rate=0.001, 
          val_check_interval=1000, 
          max_bits=1024,
          num_workers=1,
          train_ratio=0.95,
          ckpt_path=None, 
          wandb_logging=True):
    """
    Train the model and validate its performance.

    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train the model.
        num_features (int): Number of features to extract from the input signal.
        learning_rate (float): Learning rate for the optimizer.
        val_check_interval (int): Number of steps to check the validation performance.
        max_bits (int): Maximum number of bits to predict.
        num_workers (int): Number of workers for data loading.
        train_ratio (float): Ratio of training data.
        ckpt_path (str): Path to save the model checkpoints
        wandb_logging (bool): Whether to log the training process to wandb.
    """
    # Initialize Weights and Biases
    if wandb_logging:
        logger = WandbLogger(name="wahahaha_receiver", save_dir="logs")
        logger.experiment.config.update({
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        })
    else:
        logger = CSVLogger(name="wahahaha_receiver", save_dir="logs")

    # Initialize PyTorch Lightning model and datamodule and callbacks
    model = LitDenseNet(num_feats=num_features, lr=learning_rate, max_bits=max_bits)
    data_module = SignalDataModule(data_path, 
                                   batch_size=batch_size, 
                                   num_workers=num_workers, 
                                   train_ratio=train_ratio)
    best_checkpoint_callback = ModelCheckpoint(monitor="val/score", mode="max", save_top_k=5, dirpath="checkpoints", every_n_train_steps=val_check_interval, filename="{val/score:.2f}-{epoch}")
    epoch_checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_on_train_epoch_end=True, filename="end-{epoch}")


    # Define a Trainer
    trainer = L.Trainer(max_epochs=num_epochs,
                        logger=logger,
                        log_every_n_steps=4,
                        val_check_interval=val_check_interval,
                        callbacks=[
                            best_checkpoint_callback,
                            epoch_checkpoint_callback,
                        ])

    # Train the model
    print("Start training...")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Finalize the logger
    logger.finalize()

if __name__ == "__main__":
    fire.Fire(train)