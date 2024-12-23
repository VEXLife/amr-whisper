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
vocab_inv = {v: k for k, v in vocab.items()}

if __name__ == "__main__":
    # Write to vocab.json
    import json
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
