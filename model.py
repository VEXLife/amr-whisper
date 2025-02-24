from einops import rearrange
from vocab import vocab, vocab_inv
import torch
from typing import Iterable, Tuple
import numpy as np
from transformers import LogitsProcessor
import torch.nn.functional as F

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
symb_type_dict_inv = {v: k for k, v in symb_type_dict.items()}


class SignalTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.__call__ = self.encode

    def encode(self, symb_type: int, symb_wid: float, symb_seq: np.ndarray) -> torch.LongTensor:
        if symb_type == symb_type_dict_inv["<|unknownmod|>"]:
            return torch.tensor([vocab["<|unknownmod|>"], vocab["<|eos|>"]], dtype=torch.long)
        input_ids = [vocab[symb_type_dict[symb_type]]] + [vocab[f"<|{symb_wid:.2f}|>"]] + [
            vocab['0'] + symb for symb in symb_seq] + [vocab["<|eos|>"]]
        return torch.tensor(input_ids, dtype=torch.long)

    def decode(self, input_ids: torch.LongTensor) -> Tuple[int, float, list]:
        input_ids = list(input_ids[input_ids != -100])
        # print(''.join([vocab_inv[input_ids[j].item()] for j in range(len(input_ids))]))
        if vocab_inv[input_ids[0].item()] == "<|startoftranscript|>":
            input_ids = input_ids[1:]
        symb_type = symb_type_dict_inv[vocab_inv[input_ids[0].item()]]
        if symb_type == symb_type_dict_inv["<|unknownmod|>"]:
            return symb_type, 0, []
        symb_wid = float(vocab_inv[input_ids[1].item()][2:-2])
        symb_seq = [input_id.item() - vocab['0']
                    for input_id in input_ids[2:-1]]  # Filter out <|eos|>
        return symb_type, symb_wid, symb_seq

    def batch_decode(self, batch: Iterable[torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        symb_types = []
        symb_wids = []
        symb_seqs = []
        for input_ids in batch:
            symb_type, symb_wid, symb_seq = self.decode(input_ids)
            symb_types.append(symb_type)
            symb_wids.append(symb_wid)
            symb_seqs.append(symb_seq)
        return torch.tensor(symb_types, dtype=torch.long), torch.tensor(symb_wids, dtype=torch.float), symb_seqs


class SignalFeatureExtractor:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, iq_wave):
        iq_wave = torch.tensor(iq_wave, dtype=torch.float32)
        iq_wave_complex = torch.complex(iq_wave[:, 0], iq_wave[:, 1])
        spec = torch.stft(iq_wave_complex, win_length=32, n_fft=32, hop_length=8, window=torch.hann_window(32), return_complex=True)
        iq_spec = torch.cat((spec.real, spec.imag), dim=0)
        original_len = iq_spec.shape[1]
        iq_spec = rearrange(iq_spec, 'c t -> t c')
        # Pad the features
        iq_spec = torch.nn.functional.pad(
            iq_spec, (0,0,0,self.max_seq_len - original_len), mode='constant', value=0)
        return iq_spec, torch.tensor([original_len], dtype=torch.long)


class SignalLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super(SignalLogitsProcessor, self).__init__()
        self.symb_type_mask = torch.tensor(
            [vocab[symb_type_text] for symb_type_text in symb_type_dict.values()], dtype=torch.long)
        self.symb_wid_mask = torch.tensor(
            [vocab[f"<|{symb_wid:.2f}|>"] for symb_wid in torch.linspace(0, 1, 21)], dtype=torch.long)
        self.symb_seq_mask = torch.tensor(
            [i + vocab['0'] for i in range(32)] + [vocab['<|eos|>']], dtype=torch.long)

    def __call__(self, input_ids, scores):
        # The 1st token is the modulation type, the 2nd token is the symbol width, and the rest are the symbol sequence
        new_scores = torch.full_like(scores, -float('inf'))
        if input_ids.size(1) == 1:
            new_scores[:, self.symb_type_mask] = scores[:, self.symb_type_mask]
        elif input_ids.size(1) == 2:
            new_scores[:, self.symb_wid_mask] = scores[:, self.symb_wid_mask]
        else:
            new_scores[:, self.symb_seq_mask] = scores[:, self.symb_seq_mask]
        return new_scores


class ComputeMetrics:
    def __init__(self, logits_processor_list, tokenizer):
        self.logits_processor_list = logits_processor_list
        self.tokenizer = tokenizer

    def __call__(self, pred):
        pred_logits = torch.tensor(pred.predictions[0])
        pred_ids = []
        for i in range(pred_logits.shape[0]):
            pred_seq_ids = [vocab["<|startoftranscript|>"]]
            for j in range(pred_logits.shape[1]):
                pred_logits_new = self.logits_processor_list(
                    torch.tensor(pred_seq_ids).unsqueeze(0), pred_logits[i, j].unsqueeze(0))
                pred_seq_ids.append(torch.argmax(pred_logits_new))
            pred_ids.append(torch.tensor(pred_seq_ids))
        label_ids = pred.label_ids
        batch_size = pred_logits.shape[0]
        pred_symb_type, pred_symb_wid, pred_symb_seq = self.tokenizer.batch_decode(
            pred_ids)
        label_symb_type, label_symb_wid, label_symb_seq = self.tokenizer.batch_decode(
            label_ids)

        score = 0
        for i in range(batch_size):
            if label_symb_type[i] == symb_type_dict_inv["<|unknownmod|>"] or pred_symb_type[i] == symb_type_dict_inv["<|unknownmod|>"]:
                if pred_symb_type[i] == label_symb_type[i]:
                    score += 100 / batch_size
                continue

            mt_score = (pred_symb_type[i] == label_symb_type[i]) * 100

            er = torch.abs((pred_symb_wid[i] - label_symb_wid[i]) /
                           label_symb_wid[i])
            sw_score = torch.clamp(100 - (er - 0.05) / 0.15 * 100, 0, 100)

            symb_seq_hat = torch.tensor(pred_symb_seq[i], dtype=torch.float)
            symb_seq_ground_truth = torch.tensor(
                label_symb_seq[i], dtype=torch.float)
            symb_seq_ground_truth_len = symb_seq_ground_truth.numel()
            symb_seq_hat_len = symb_seq_hat.numel()

            if symb_seq_ground_truth_len == 0 or symb_seq_hat_len == 0:
                continue
            symb_seq_hat = F.pad(
                symb_seq_hat, (0, symb_seq_ground_truth_len - symb_seq_hat_len), "constant", 0)

            cs = torch.cosine_similarity(
                symb_seq_hat.unsqueeze(0), symb_seq_ground_truth.unsqueeze(0)
            )
            cq_score = torch.clamp((cs - 0.7) / 0.25 *
                                    100, 0, 100)
            
            score += (0.2 * mt_score + 0.3 * sw_score + 0.5 * cq_score) / batch_size

        return {
            "score": score,
        }
