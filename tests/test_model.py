import pytest
import torch
import torch.nn.functional as F
import numpy as np
from transformers import LogitsProcessorList
import sys
sys.path.append(".")
from vocab import vocab, vocab_inv
from model import SignalTokenizer, SignalFeatureExtractor, SignalLogitsProcessor, ComputeMetrics, symb_type_dict, symb_type_dict_inv


@pytest.fixture
def tokenizer():
    return SignalTokenizer(vocab)


@pytest.fixture
def feature_extractor():
    return SignalFeatureExtractor(max_seq_len=1024)


@pytest.fixture
def logits_processor():
    return SignalLogitsProcessor()


@pytest.fixture
def logits_processor_list(logits_processor):
    return LogitsProcessorList([logits_processor])


@pytest.fixture
def compute_metrics(logits_processor_list, tokenizer):
    return ComputeMetrics(logits_processor_list, tokenizer)


def test_tokenizer_encode_decode(tokenizer):
    # Test normal case
    symb_type = symb_type_dict_inv["<|16QAM|>"]
    symb_wid = 0.25
    symb_seq = np.array([1, 2, 3, 0])
    encoded = tokenizer.encode(symb_type, symb_wid, symb_seq)

    decoded_type, decoded_wid, decoded_seq = tokenizer.decode(encoded)
    assert decoded_type == symb_type
    assert decoded_wid == pytest.approx(symb_wid, 0.01)
    assert decoded_seq == symb_seq.tolist()

    # Test unknown modulation
    unknown_encoded = tokenizer.encode(
        symb_type_dict_inv["<|unknownmod|>"], 0, np.array([]))
    decoded_type, _, _ = tokenizer.decode(unknown_encoded)
    assert decoded_type == symb_type_dict_inv["<|unknownmod|>"]


def test_logits_processor(logits_processor):
    # Test first token masking
    input_ids = torch.tensor([[vocab["<|startoftranscript|>"]]])
    scores = torch.randn(1, len(vocab))
    processed = logits_processor(input_ids, scores)
    assert (processed[:, logits_processor.symb_type_mask]
            > -float('inf')).all()

    # Test second token masking
    input_ids = torch.tensor(
        [[vocab["<|startoftranscript|>"], vocab[symb_type_dict[symb_type_dict_inv["<|16QAM|>"]]]]])
    scores = torch.randn(1, len(vocab))
    processed = logits_processor(input_ids, scores)
    assert (processed[:, logits_processor.symb_wid_mask] > -float('inf')).all()

    # Test third token masking to see if eos is masked
    input_ids = torch.tensor([[vocab["<|startoftranscript|>"],
                             vocab[symb_type_dict[symb_type_dict_inv["<|16QAM|>"]]], vocab["<|0.25|>"]]])
    scores = torch.randn(1, len(vocab))
    processed = logits_processor(input_ids, scores)
    assert (processed[:, vocab["<|eos|>"]] > -float('inf')).all()


def test_compute_metrics(tokenizer, compute_metrics):
    # Mock prediction data
    class MockPred:
        prediction_ids = torch.stack([
            tokenizer.encode(
                symb_type_dict_inv["<|16QAM|>"], 0.25, np.array([1, 2, 3, 0])),
            torch.concat([
                tokenizer.encode(
                    symb_type_dict_inv["<|unknownmod|>"], 0, np.array([1, 2, 3, 0])),
                torch.ones(5, dtype=torch.long) * 0
            ])
        ])
        predictions = F.one_hot(
            prediction_ids, num_classes=len(vocab)).float().unsqueeze(0)

        label_ids = torch.stack([
            tokenizer.encode(
                symb_type_dict_inv["<|16QAM|>"], 0.25, np.array([1, 2, 3, 0])),
            torch.concat([
                tokenizer.encode(
                    symb_type_dict_inv["<|unknownmod|>"], 0, np.array([1, 2, 3, 0])),
                torch.ones(5, dtype=torch.long) * -100
            ])
        ])

    metrics = compute_metrics(MockPred())
    assert metrics["score"] == 100

    class MockPred2:
        prediction_ids = torch.stack([
            tokenizer.encode(
                symb_type_dict_inv["<|8QAM|>"], 0.35, np.array([1, 2, 5, 0])),
            tokenizer.encode(
                symb_type_dict_inv["<|8PSK|>"], 0.25, np.array([1, 2, 3, 0])),
        ])
        predictions = F.one_hot(
            prediction_ids, num_classes=len(vocab)).float().unsqueeze(0)

        label_ids = torch.stack([
            tokenizer.encode(
                symb_type_dict_inv["<|16QAM|>"], 0.25, np.array([1, 2, 3, 0])),
            torch.concat([
                tokenizer.encode(
                    symb_type_dict_inv["<|unknownmod|>"], 0, np.array([1, 2, 3, 0])),
                torch.ones(5, dtype=torch.long) * -100
            ])
        ])

    metrics = compute_metrics(MockPred2())
    assert metrics["score"] == 25

    class MockPred3:
        prediction_ids = torch.stack([
            tokenizer.encode(
                symb_type_dict_inv["<|unknownmod|>"], 0, np.array([])),
        ])
        predictions = F.one_hot(
            prediction_ids, num_classes=len(vocab)).float().unsqueeze(0)

        label_ids = torch.stack([
            torch.concat([
                tokenizer.encode(
                    symb_type_dict_inv["<|MSK|>"], 0.65, np.array([1, 2, 3, 0])),
                torch.ones(2, dtype=torch.long) * -100
            ])
        ])

    metrics = compute_metrics(MockPred3())
    assert metrics["score"] == 0


# Additional edge case tests
def test_empty_sequence(tokenizer):
    encoded = tokenizer.encode(11, 0, np.array([]))
    assert encoded.tolist() == [vocab["<|unknownmod|>"], vocab["<|eos|>"]]


test_compute_metrics(
    SignalTokenizer(vocab),
    ComputeMetrics(
        LogitsProcessorList([SignalLogitsProcessor()]),
        SignalTokenizer(vocab)
    )
)
