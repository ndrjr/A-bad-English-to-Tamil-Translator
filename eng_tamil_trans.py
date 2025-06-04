import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os

# Load data
df = pd.read_parquet('Downloads/wiki_train.parquet')  # Fixed path separator
df.head()

x = df['eng_Latn'].values
y = df['tam_Taml'].values

x = x[:20000]
y = y[:20000]

# Save training data for tokenizer training
with open("tam.txt", "w", encoding="utf-8") as f:
    for line in y:
        if line is not None:  # Check for None values
            f.write(str(line).strip() + "\n")

with open("eng.txt", "w", encoding="utf-8") as f:
    for line in x:
        if line is not None:  # Check for None values
            f.write(str(line).strip() + "\n")

# Train tokenizers
# English
spm.SentencePieceTrainer.train(input='eng.txt', model_prefix='eng_tokenizer', vocab_size=8000)

# Tamil
spm.SentencePieceTrainer.train(input='tam.txt', model_prefix='tam_tokenizer', vocab_size=8000)

# Load tokenizers
eng_sp = spm.SentencePieceProcessor(model_file='eng_tokenizer.model')
tam_sp = spm.SentencePieceProcessor(model_file='tam_tokenizer.model')

# Test tokenizers
print("English tokenization:", eng_sp.encode("Let me break it down", out_type=int))
print("Tamil tokenization:", tam_sp.encode("நான் படிக்கிறேன்", out_type=int))

class TranslationDataset(Dataset):
    def __init__(self, eng_sentences, tam_sentences, eng_tokenizer, tam_tokenizer, max_len=50):
        self.eng = eng_sentences
        self.tam = tam_sentences
        self.eng_tok = eng_tokenizer
        self.tam_tok = tam_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.eng)

    def __getitem__(self, idx):
        eng_sentence = str(self.eng[idx]) if self.eng[idx] is not None else ""
        tam_sentence = str(self.tam[idx]) if self.tam[idx] is not None else ""

        src_ids = self.eng_tok.encode(eng_sentence, out_type=int)
        tgt_ids = self.tam_tok.encode(tam_sentence, out_type=int)

        # Truncate to max_len
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len - 1]  # Leave space for EOS

        # Prepare decoder input and target
        decoder_input = [1] + tgt_ids  # BOS + target
        decoder_target = tgt_ids + [2]  # target + EOS

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(decoder_input, dtype=torch.long),
            "tgt_out": torch.tensor(decoder_target, dtype=torch.long)
        }

def collate_fn(batch):
    src_batch = [item["src"] for item in batch]
    tgt_in_batch = [item["tgt_in"] for item in batch]
    tgt_out_batch = [item["tgt_out"] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_in_padded = pad_sequence(tgt_in_batch, batch_first=True, padding_value=0)
    tgt_out_padded = pad_sequence(tgt_out_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_in_padded, tgt_out_padded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # Handle odd d_model
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        # Linear projections and reshape
        Q = self.q_linear(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Expand mask to match the shape [batch_size, num_heads, query_len, key_len]
            if mask.dim() == 2:  # [query_len, key_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_len, key_len]
            elif mask.dim() == 3:  # [batch_size, query_len, key_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, query_len, key_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        return self.out_linear(concat)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = AddNorm(d_model, dropout)
        self.norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x, self.attn(x, x, x, mask))
        x = self.norm2(x, self.ff(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = AddNorm(d_model, dropout)
        self.norm2 = AddNorm(d_model, dropout)
        self.norm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked self-attention
        x = self.norm1(x, self.self_attn(x, x, x, tgt_mask))

        # Encoder-decoder attention
        x = self.norm2(x, self.enc_dec_attn(x, enc_output, enc_output, src_mask))

        # Feed forward
        x = self.norm3(x, self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for block in self.blocks:
            x = block(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for block in self.blocks:
            x = block(x, enc_output, tgt_mask, src_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_layers=4, num_heads=8, d_ff=512, dropout=0.1, max_len=100):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt_in, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt_in, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with 0."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return ~mask  # Invert: True where allowed, False where masked

def create_padding_mask(seq, pad_idx=0):
    """Create padding mask where pad_idx positions are masked (False)."""
    return (seq != pad_idx)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create dataset and dataloader
dataset = TranslationDataset(x, y, eng_sp, tam_sp, max_len=50)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = Transformer(
    src_vocab=8000, tgt_vocab=8000, d_model=256,
    num_layers=4, num_heads=8, d_ff=512
).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

def train_model(model, dataloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (src, tgt_in, tgt_out) in enumerate(loop):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            # Create masks
            tgt_seq_len = tgt_in.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            # Forward pass
            output = model(src, tgt_in, src_mask=None, tgt_mask=tgt_mask)

            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            tgt_out = tgt_out.view(-1)

            # Calculate loss
            loss = criterion(output, tgt_out)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

def translate(model, sentence, eng_tokenizer, tam_tokenizer, max_len=50, device="cpu"):
    model.eval()

    # Tokenize source sentence
    src_ids = eng_tokenizer.encode(sentence, out_type=int)[:max_len]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Start with BOS token
    tgt_ids = [1]  # BOS token

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

            # Forward pass
            output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)

            # Get next token
            next_token = output[0, -1].argmax(-1).item()

            if next_token == 2:  # EOS token
                break

            tgt_ids.append(next_token)

    # Decode (skip BOS token)
    translated_text = tam_tokenizer.decode(tgt_ids[1:])
    return translated_text

# Train the model
print("Starting training...")
train_model(model, loader, epochs=10)

# Test translation
print("\nTesting translation...")
model.to(device)

test_sentences = [
    "Get out",
    "Hello, how are you?",
    "I am learning Tamil"
]

for sentence in test_sentences:
    translated = translate(model, sentence, eng_sp, tam_sp, device=device)
    print(f"English: {sentence}")
    print(f"Tamil: {translated}")
    print("-" * 50)

# Save the model
torch.save(model.state_dict(), 'translation_model.pth')
print("Model saved as 'translation_model.pth'")
