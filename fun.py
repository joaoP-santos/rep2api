import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, text_vocab_size, gloss_vocab_size, embedding_dim, nhead, num_encoder_layers, num_decoder_layers, dropout, max_len):
        super().__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim)
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_dim, gloss_vocab_size)

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        if sz == 0: # Handle the case where sz is 0
            return torch.empty(0, 0)  # Return an empty tensor if sz is 0
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq):
        # Create a padding mask of shape (batch_size, seq_len)
        return seq == 0 # text_word_to_index["<pad>"]

    def forward(self, src, tgt, inference=False):    
        # Source embeddings remain the same.
        src_embedded = self.positional_encoding(self.text_embedding(src))
        src_padding_mask = self.create_padding_mask(src)
    
        # For training, use teacher forcing by slicing the target.
        if not inference:
            # Use all tokens except the final token
            tgt_input = tgt[:, :-1]
            tgt_padding_mask = self.create_padding_mask(tgt_input)
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
            tgt_embedded = self.positional_encoding(self.gloss_embedding(tgt_input))
        else:
            # In inference mode, use the entire target sequence so far.
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
            tgt_embedded = self.positional_encoding(self.gloss_embedding(tgt))
    
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.fc_out(output)   

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):  # Increased max_len from 100 to 4096
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Apply positional encoding to sequence length
        return self.dropout(x)


import re
import string

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join(text.split())  # Remove extra whiatespace
    return text

def translate_text_to_gloss(model, input_text, text_word_to_index, gloss_word_to_index, gloss_index_to_word, device, max_len=100):
    model.eval()
    processed_text = preprocess_text(input_text)
    input_tokens = [text_word_to_index.get(token, text_word_to_index["<unk>"]) for token in processed_text.split()]
    input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)
    
    # Start with <start> token
    output_tokens = [gloss_word_to_index["<start>"]]
    
    for _ in range(max_len):
        target_tensor = torch.tensor(output_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            # Call forward in inference mode so the full sequence is used.
            output = model(input_tensor, target_tensor, inference=True)
        
        # The model outputs one logit per token in the target sequence.
        # We take the logits corresponding to the last token.
        if output.shape[1] == 0:
            print("Warning: Output sequence length is 0. Stopping decoding.")
            break

        predicted_token_index = output[:, -1].argmax(-1).item()
        output_tokens.append(predicted_token_index)
        
        if predicted_token_index == gloss_word_to_index["<end>"]:
            break

    # Remove <start> and <end> tokens for display.
    predicted_gloss_words = [gloss_index_to_word[token] for token in output_tokens if token not in (gloss_word_to_index["<start>"], gloss_word_to_index["<end>"])]
    return " ".join(predicted_gloss_words)

print('loaded fun.py')