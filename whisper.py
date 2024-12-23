#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import WhisperForConditionalGeneration, WhisperConfig
import torch

# Create a vocab json including modulation types and 0~31 modulation indices
vocab = {
    "<|eos|>": 0,
    "<|startoftranscript|>": 1,
    "<|unk|>": 2,
    "<|pad|>": 3,
    "<|cls|>": 4,
}
vocab_len = len(vocab)
added_tokens = ["<|BPSK|>", "<|QPSK|>", "<|8PSK|>", "<|MSK|>", "<|8QAM|>", "<|16QAM|>", "<|32QAM|>", "<|8APSK|>", "<|16APSK|>", "<|32APSK|>", "<|unknownmod|>"]
for symb_wid in torch.linspace(0,1,21):
    added_tokens.append(f"<|{symb_wid:.2f}|>")
for added_token in added_tokens:
    vocab[added_token] = vocab_len
    vocab_len += 1
vocab_len = len(vocab)
for i in range(32):
    ch = chr(i + ord('0'))
    vocab[ch] = vocab_len
    vocab_len += 1

# Write to vocab.json
import json
with open("vocab.json", "w") as f:
    json.dump(vocab, f)


# In[2]:


model_config = WhisperConfig(
    vocab_size=vocab_len,
    num_mel_bins=2,
    max_source_positions=1024,
    pad_token_id=vocab["<|pad|>"],
    bos_token_id=vocab["<|startoftranscript|>"],
    eos_token_id=vocab["<|eos|>"],
    decoder_start_token_id=vocab["<|startoftranscript|>"],
)
model = WhisperForConditionalGeneration(config=model_config)


# In[3]:


from torch.utils.data import Dataset
import glob
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence as rnn_utils
from einops import rearrange
import numpy as np
from typing import List, Tuple

symb_type_dict = {
    1: "<|BPSK|>",
    2: "<|QPSK|>",
    3: "<|8PSK|>",
    4: "<|MSK|>",
    5: "<|8QAM|>",
    6: "<|16QAM|>",
    7: "<|32QAM|>",
    8: "<|8APSK|>",
    9: "<|16APSK|>",
    10: "<|32APSK|>",
    11: "<|unknownmod|>"
}
vocab_inv = {v: k for k, v in vocab.items()}
symb_type_dict_inv = {v: k for k, v in symb_type_dict.items()}

class SignalTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.__call__ = self.encode

    def encode(self, symb_type: int, symb_wid: float, symb_seq: np.ndarray) -> torch.LongTensor:
        input_ids = [vocab[symb_type_dict[symb_type]]] + [vocab[f"<|{symb_wid:.2f}|>"]] + [vocab['0'] + symb for symb in symb_seq] + [vocab["<|eos|>"]]
        return torch.tensor(input_ids, dtype=torch.long)
    
    def decode(self, input_ids: torch.LongTensor) -> Tuple[int, float, list]:
        input_ids = list(input_ids[input_ids != -100])
        # print(''.join([vocab_inv[input_ids[j].item()] for j in range(len(input_ids))]))
        if vocab_inv[input_ids[0].item()] == "<|startoftranscript|>":
            input_ids = input_ids[1:]
        symb_type = symb_type_dict_inv[vocab_inv[input_ids[0].item()]]
        symb_wid = float(vocab_inv[input_ids[1].item()][2:-2])
        symb_seq = [input_id.item() - vocab['0'] for input_id in input_ids[2:]]
        return symb_type, symb_wid, symb_seq

    def batch_decode(self, batch: torch.LongTensor | List[torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        symb_types = []
        symb_wids = []
        symb_seqs = []
        for input_ids in batch:
            symb_type, symb_wid, symb_seq = self.decode(input_ids)
            symb_types.append(symb_type)
            symb_wids.append(symb_wid)
            symb_seqs.append(symb_seq)
        return torch.tensor(symb_types, dtype=torch.long), torch.tensor(symb_wids, dtype=torch.float), symb_seqs


# In[4]:


tokenizer = SignalTokenizer(vocab)
tok=tokenizer.encode(1, 0.5, np.array([0,9,2,7]))
print(tok)
print(tokenizer.decode(tok))


# In[5]:


class SignalDataset(Dataset):
    def __init__(self, data_path):
        super(SignalDataset, self).__init__()
        # Recursively find all csv files in the data_path
        self.file_list = glob.glob(os.path.join(data_path, '**/*.csv'), recursive=True)
        self.cache = {}  # Dictionary for caching data

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        
        data = pd.read_csv(self.file_list[index], header=None, names=['I', 'Q', 'Code Sequence', 'Modulation Type', 'Symbol Width'])
        
        iq_wave = data[['I', 'Q']].values
        symb_seq = data['Code Sequence'].dropna().astype(int).values
        symb_type = data['Modulation Type'].values[0]
        symb_wid = data['Symbol Width'].values[0]

        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        iq_wave = rearrange(iq_wave, 't c -> c t')
        # Pad the features to 2048
        iq_wave = torch.nn.functional.pad(iq_wave, (0, 2048 - iq_wave.shape[1]), mode='constant', value=0)

        target = tokenizer.encode(symb_type, symb_wid, symb_seq)
        # Cache processed data
        self.cache[index] = (iq_wave, target)

        return iq_wave, target
    
def _collator_fn(batch):
    input_features = rnn_utils([item[0] for item in batch], batch_first=True)
    labels = rnn_utils([item[1] for item in batch], batch_first=True, padding_value=-100)
    return {
        "input_features": input_features,
        "labels": labels,
    }


# In[6]:


from transformers import LogitsProcessor, LogitsProcessorList

class SignalLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super(SignalLogitsProcessor, self).__init__()
        self.symb_type_mask = torch.tensor([vocab[symb_type_text] for symb_type_text in symb_type_dict.values()], dtype=torch.long)
        self.symb_wid_mask = torch.tensor([vocab[f"<|{symb_wid:.2f}|>"] for symb_wid in torch.linspace(0,1,21)], dtype=torch.long)
        self.symb_seq_mask = torch.tensor([i + vocab['0'] for i in range(32)], dtype=torch.long)

    def __call__(self, input_ids, scores):
        # The 1st token is the modulation type, the 2nd token is the symbol width, and the rest are the symbol sequence
        new_scores = torch.full_like(scores, -float('inf'))
        if input_ids.numel() == 1:
            new_scores[:, self.symb_type_mask] = scores[:, self.symb_type_mask]
        elif input_ids.numel() == 2:
            new_scores[:, self.symb_wid_mask] = scores[:, self.symb_wid_mask]
        else:
            new_scores[:, self.symb_seq_mask] = scores[:, self.symb_seq_mask]
        return new_scores


# In[7]:


logits_processor_list = LogitsProcessorList([SignalLogitsProcessor()])
tok=model.generate(input_features=torch.randn(1, 2, 2048), logits_processor=logits_processor_list)
print(tokenizer.batch_decode(tok))


# In[8]:


# Split the dataset into training and validation sets
from torch.utils.data import random_split
dataset = SignalDataset('/root/autodl-tmp/train_data')
train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset[0]


# In[9]:




# In[ ]:


# Finetune Whisper on dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/whisper_iq",
    run_name="whisper_finetune",
    learning_rate=1e-4,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=5000,
    logging_dir="./logs/whisper_iq",
    logging_steps=100,
    metric_for_best_model="score",
    save_strategy="steps",
    save_steps=1000,
    report_to="wandb",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=_collator_fn,
    compute_metrics=compute_metrics,
)
trainer.train()


# In[14]:


output_ids = model.generate(dataset[0][0].unsqueeze(0))
tokenizer.batch_decode(output_ids)

