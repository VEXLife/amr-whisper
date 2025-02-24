from wenet.transformer.encoder import ConformerEncoder
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional
import math

class AEDConfig(PretrainedConfig):
    model_type = "Conformer-Transformer"

    def __init__(self,
                 mel_bins,
                 vocab_size,
                 d_model=256,
                 encoder_layers=6,
                 decoder_layers=6,
                 encoder_head=4,
                 decoder_head=4,
                 encoder_ffn_dim=1024,
                 decoder_ffn_dim=1024,
                 max_seq_len=5000,
                 pad=0,
                 sos=1,
                 eos=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.mel_bins = mel_bins
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_head = encoder_head
        self.decoder_head = decoder_head
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.max_seq_len = max_seq_len
        self.pad = pad
        self.sos = sos
        self.eos = eos

class AED(PreTrainedModel):
    config_class = AEDConfig

    def __init__(self, config: AEDConfig):
        super().__init__(config)
        self.encoder = ConformerEncoder(
            input_size=config.mel_bins, 
            output_size=config.d_model, 
            attention_heads=config.encoder_head,
            linear_units=config.encoder_ffn_dim,
            num_blocks=config.encoder_layers,
        )
      
        # Decoder components
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_len)
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_head,
            dim_feedforward=config.decoder_ffn_dim,
            batch_first=True  # Ensure compatibility with Conformer's output
        )
        self.decoder = TransformerDecoder(decoder_layer, config.decoder_layers)
      
        self.proj = nn.Linear(config.d_model, config.vocab_size)
        self.max_seq_len = config.max_seq_len
        self.pad = config.pad
        self.sos = config.sos
        self.eos = config.eos

    def forward(
        self, 
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        # Encoder processing
        encoder_outputs, encoder_mask = self.encoder(inputs, input_lengths)
      
        # Convert target indices to embeddings
        tgt = self.embedding(targets)
        tgt = self.pos_encoder(tgt)
      
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(targets.device)
        tgt_key_padding_mask = self.create_padding_mask(targets, target_lengths)
      
        # Decoder processing
        decoder_output = self.decoder(
            tgt=tgt,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=encoder_mask.unsqueeze(1).bool(),  # Adjust mask shape
        )
      
        return self.proj(decoder_output)
    
    def generate(
            self, 
            input_features: torch.Tensor,
            input_lengths: torch.Tensor,
            max_length: Optional[int] = None
        ) -> torch.Tensor:
        max_length = max_length or self.max_seq_len
        batch_size = input_features.size(0)
    
        # Encode input features
        encoder_outputs, encoder_mask = self.encoder(input_features, input_lengths)
    
        # 修复1：正确转换mask维度 [batch, seq_len]
        encoder_mask = encoder_mask.squeeze(1) if encoder_mask.dim() == 3 else encoder_mask
        encoder_key_padding_mask = encoder_mask.bool()  # [batch, seq_len]
    
        # Initialize with SOS token
        decoder_input = torch.full((batch_size, 1), self.sos, dtype=torch.long, device=input_features.device)
    
        for _ in range(max_length):
            # Embed and add positional encoding
            embedded = self.embedding(decoder_input)
            embedded = self.pos_encoder(embedded)
        
            # Target mask
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(decoder_input.device)
        
            # Decode step
            output = self.decoder(
                tgt=embedded,
                memory=encoder_outputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=encoder_key_padding_mask,
            )
        
            # Get next token
            logits = self.proj(output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
            if (next_token == self.eos).all():
                break
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
    
        return decoder_input

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def create_padding_mask(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_len = sequences.size()
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)