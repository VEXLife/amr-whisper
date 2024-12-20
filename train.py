import fire
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from dataset import SignalDataModule
from model import LitDenseNet

def train(*, data_path, batch_size=32, num_epochs=20, num_features=150, learning_rate=0.001, ckpt_path=None):
    """
    Train the model and validate its performance.

    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train the model.
        num_features (int): Number of features to extract from the input signal.
        learning_rate (float): Learning rate for the optimizer.
        ckpt_path (str): Path to save the model checkpoints
    """
    # Initialize Weights and Biases
    logger = WandbLogger(name="wahahaha_receiver", save_dir="logs")
    logger.experiment.config.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    })

    # Initialize PyTorch Lightning model and datamodule
    model = LitDenseNet(num_feats=num_features, lr=learning_rate)
    data_module = SignalDataModule(data_path, batch_size=batch_size)

    # Define a Trainer
    trainer = L.Trainer(max_epochs=num_epochs,
                        logger=logger,
                        log_every_n_steps=4)

    # Train the model
    print("Start training...")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Finalize the logger
    logger.finalize()

if __name__ == "__main__":
    fire.Fire(train)