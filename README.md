# 第三届电磁大数据竞赛代码库

赛题可视为由电磁信号预测调制类型、码元宽度和解调码序列的机器学习任务。我们认为该问题与语音识别问题等价，故仅通过修改Whisper模型的词汇表，去掉梅尔倒频谱前处理部分，不加额外处理地将信号IQ数据直接经过STFT后送入Whisper模型进行训练和推理。具体参见代码。

本README不提供其他语种，因为赛题只提供了中文版本。

## 环境配置

请使用 `uv` 或 `poetry` 等支持 `pyproject.toml` 的工具来安装依赖。

```bash
uv sync
```

## 训练

仓库使用fire解析命令行参数，使用 `python train.py -- --help` 查看帮助信息。

您应看到如下帮助信息：
```bash
NAME
    train.py - Trains a Whisper model for conditional generation on a given dataset.

SYNOPSIS
    train.py <flags>

DESCRIPTION
    Trains a Whisper model for conditional generation on a given dataset.

FLAGS
    --learning_rate=LEARNING_RATE
        Default: 0.0001
        The learning rate for training. Defaults to 1e-4.
    --num_train_epochs=NUM_TRAIN_EPOCHS
        Default: 20
        The number of training epochs. Defaults to 20.
    --per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE
        Default: 16
        Batch size per device during training. Defaults to 16.
    --per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE
        Default: 16
        Batch size per device during evaluation. Defaults to 16.
    -w, --weight_decay=WEIGHT_DECAY
        Default: 0.01
        Weight decay for optimization. Defaults to 0.01.
    --eval_steps=EVAL_STEPS
        Default: 5000
        Number of steps between evaluations. Defaults to 5000.
    --logging_steps=LOGGING_STEPS
        Default: 100
        Number of steps between logging. Defaults to 100.
    --save_steps=SAVE_STEPS
        Default: 1000
        Number of steps between model checkpoints. Defaults to 1000.
    -o, --output_dir=OUTPUT_DIR
        Default: './logs/whisper_iq'
        Directory to save model checkpoints. Defaults to "./logs/whisper_iq".
    --logging_dir=LOGGING_DIR
        Default: './logs/whisper_iq'
        Directory to save logs. Defaults to "./logs/whisper_iq".
    --run_name=RUN_NAME
        Default: 'whisper_finetune'
        Name of the training run. Defaults to "whisper_finetune".
    --report_to=REPORT_TO
        Default: 'tensorboard'
        Reporting tool for logging (e.g., "tensorboard"). Defaults to "tensorboard".
    --dataset_path=DATASET_PATH
        Default: './train_data'
        Path to the training dataset. Defaults to './train_data'.
    -t, --train_ratio=TRAIN_RATIO
        Default: 0.99
        Ratio of the dataset to use for training. Defaults to 0.99.
    -m, --max_seq_len=MAX_SEQ_LEN
        Default: 2048
        Maximum number of sequences. Defaults to 2048.
    --encoder_attention_heads=ENCODER_ATTENTION_HEADS
        Default: 6
        Number of attention heads in the encoder. Defaults to 6.
    --encoder_ffn_dim=ENCODER_FFN_DIM
        Default: 1536
        Hidden layer size in the encoder. Defaults to 1536.
    --encoder_layers=ENCODER_LAYERS
        Default: 4
        Number of layers in the encoder. Defaults to 4.
    --decoder_attention_heads=DECODER_ATTENTION_HEADS
        Default: 6
        Number of attention heads in the decoder. Defaults to 6.
    --decoder_ffn_dim=DECODER_FFN_DIM
        Default: 1536
        Hidden layer size in the decoder. Defaults to 1536.
    --decoder_layers=DECODER_LAYERS
        Default: 4
        Number of layers in the decoder. Defaults to 4.
    --d_model=D_MODEL
        Default: 384
        Hidden dimension of the model. Defaults to 384.
    -a, --attn_implementation=ATTN_IMPLEMENTATION
        Default: 'flash_attention_2'
        Attention implementation to use. Can be 'eager', 'sdpa' or 'flash_attention_2'. Defaults to "flash_attention_2".
    --save_total_limit=SAVE_TOTAL_LIMIT
        Default: 20
        Maximum number of checkpoints to keep. Defaults to 20.
    --resume_from_checkpoint=RESUME_FROM_CHECKPOINT
        Type: Optional[]
        Default: None
        Resume training from given checkpoint dir.
    --num_workers=NUM_WORKERS
        Default: 4
        Number of workers for data loading. Defaults to 4.
```

训练时，我们的超参数与代码中的默认值一致。注意路径需要根据您的情况修改。

## 运行

运行方法与官方赛题要求一致，为

```bash
python run.py <测试目录> <csv结果文件保存路径>
```
