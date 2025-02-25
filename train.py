import fire
from torch.utils.data import random_split
from transformers import (LogitsProcessorList, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, WhisperConfig,
                          WhisperForConditionalGeneration)

from dataset import SignalDataset, collator_fn
from model import ComputeMetrics, SignalLogitsProcessor, SignalTokenizer, SignalFeatureExtractor
from vocab import vocab, vocab_inv, vocab_len


def train(learning_rate=1e-4, num_train_epochs=20, per_device_train_batch_size=16,
          per_device_eval_batch_size=16, weight_decay=0.01, eval_steps=5000,
          logging_steps=100, save_steps=1000, output_dir="./logs/whisper_iq",
          logging_dir="./logs/whisper_iq", run_name="whisper_finetune", report_to="tensorboard",
          dataset_path='./train_data', train_ratio=0.99, max_seq_len=2048,
          encoder_attention_heads=6, encoder_ffn_dim=1536, encoder_layers=4,
          decoder_attention_heads=6, decoder_ffn_dim=1536, decoder_layers=4,
          d_model=384, attn_implementation="flash_attention_2",
          save_total_limit=20, resume_from_checkpoint=None, num_workers=4):
    """
    Trains a Whisper model for conditional generation on a given dataset.

    Args:
        learning_rate (float, optional): The learning rate for training. Defaults to 1e-4.
        num_train_epochs (int, optional): The number of training epochs. Defaults to 20.
        per_device_train_batch_size (int, optional): Batch size per device during training. Defaults to 16.
        per_device_eval_batch_size (int, optional): Batch size per device during evaluation. Defaults to 16.
        weight_decay (float, optional): Weight decay for optimization. Defaults to 0.01.
        eval_steps (int, optional): Number of steps between evaluations. Defaults to 5000.
        logging_steps (int, optional): Number of steps between logging. Defaults to 100.
        save_steps (int, optional): Number of steps between model checkpoints. Defaults to 1000.
        output_dir (str, optional): Directory to save model checkpoints. Defaults to "./logs/whisper_iq".
        logging_dir (str, optional): Directory to save logs. Defaults to "./logs/whisper_iq".
        run_name (str, optional): Name of the training run. Defaults to "whisper_finetune".
        report_to (str, optional): Reporting tool for logging (e.g., "tensorboard"). Defaults to "tensorboard".
        dataset_path (str, optional): Path to the training dataset. Defaults to './train_data'.
        train_ratio (float, optional): Ratio of the dataset to use for training. Defaults to 0.99.
        max_seq_len (int, optional): Maximum number of sequences. Defaults to 2048.
        encoder_attention_heads (int, optional): Number of attention heads in the encoder. Defaults to 6.
        encoder_ffn_dim (int, optional): Hidden layer size in the encoder. Defaults to 1536.
        encoder_layers (int, optional): Number of layers in the encoder. Defaults to 4.
        decoder_attention_heads (int, optional): Number of attention heads in the decoder. Defaults to 6.
        decoder_ffn_dim (int, optional): Hidden layer size in the decoder. Defaults to 1536.
        decoder_layers (int, optional): Number of layers in the decoder. Defaults to 4.
        d_model (int, optional): Hidden dimension of the model. Defaults to 384.
        attn_implementation (str, optional): Attention implementation to use. Can be 'eager', 'sdpa' or 'flash_attention_2'. Defaults to "flash_attention_2".
        save_total_limit (int, optional): Maximum number of checkpoints to keep. Defaults to 20.
        resume_from_checkpoint (str, bool, optional): Resume training from given checkpoint dir.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        
    Returns:
        None
    """
    if resume_from_checkpoint is not None:
        model = WhisperForConditionalGeneration.from_pretrained(resume_from_checkpoint,
            attn_implementation=attn_implementation)
    else:
        model_config = WhisperConfig(
            vocab_size=vocab_len,
            num_mel_bins=64,
            max_source_positions=max_seq_len // 2, # Divide by 2 because the second conv in Whisper has a stride of 2
            pad_token_id=vocab["<|pad|>"],
            bos_token_id=vocab["<|startoftranscript|>"],
            eos_token_id=vocab["<|eos|>"],
            decoder_start_token_id=vocab["<|startoftranscript|>"],
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
            d_model=d_model,
            attn_implementation=attn_implementation,
        )
        model = WhisperForConditionalGeneration(config=model_config)

    tokenizer = SignalTokenizer(vocab)
    feature_extractor = SignalFeatureExtractor(max_seq_len)
    dataset = SignalDataset(dataset_path, feature_extractor, tokenizer)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logits_processor = SignalLogitsProcessor()
    logits_processor_list = LogitsProcessorList([logits_processor])
    compute_metrics = ComputeMetrics(logits_processor_list, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        metric_for_best_model="score",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    fire.Fire(train)
