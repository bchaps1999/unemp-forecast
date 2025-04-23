import torch
import torch.nn as nn
import math

# --- Custom Positional Embedding Layer Definition (PyTorch) ---
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Compute positional encoding matrix once during initialization
        pos_encoding = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's part of the state_dict but not trained
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0)) # Add batch dim

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # Add positional encoding up to the length of the input sequence
        # Slicing handles sequences potentially shorter than max seq_len if needed
        return x + self.pos_encoding[:, :x.size(1), :]

# --- Transformer Model Building Functions (PyTorch) ---
class TransformerEncoderBlock(nn.Module):
    # "\"\"Creates a single Transformer Encoder block in PyTorch.\"\"\"
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        # Multi-Head Attention
        # batch_first=True expects input shape (batch, seq, feature)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_mha = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch, seq_len, embed_dim)
        # src_key_padding_mask shape: (batch, seq_len), True where padded

        # Multi-Head Attention + Residual
        attn_output, _ = self.mha(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = x + self.dropout_mha(attn_output)
        x = self.norm1(x)

        # Feed Forward + Residual
        ffn_output = self.ffn(x)
        x = x + self.dropout_ffn(ffn_output)
        x = self.norm2(x)
        return x

class TransformerForecastingModel(nn.Module):
     # "\"\"Complete Transformer forecasting model in PyTorch.\"\"\"
    def __init__(self, input_dim, seq_len, embed_dim, num_heads, ff_dim,
                 num_transformer_blocks, mlp_units, dropout, mlp_dropout,
                 n_classes):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.relu = nn.ReLU() # Added ReLU after projection

        # Positional Embedding
        self.pos_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.layernorm_embed = nn.LayerNorm(embed_dim) # Added LayerNorm after positional embedding
        self.input_dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # Pooling layer (using mean pooling over sequence dim)
        # We'll apply this manually in forward after handling mask

        # Final MLP Head
        mlp_layers = []
        current_dim = embed_dim # Input to MLP is pooled transformer output
        for i, units in enumerate(mlp_units):
            mlp_layers.append(nn.Linear(current_dim, units))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            current_dim = units
        self.mlp_head = nn.Sequential(*mlp_layers)

        # Output layer for classification
        self.output_layer = nn.Linear(current_dim, n_classes) # Output logits

    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch, seq_len, input_dim)
        # src_key_padding_mask shape: (batch, seq_len), True where padded

        # Input projection
        x = self.relu(self.input_projection(x)) # (batch, seq_len, embed_dim)

        # Add positional embedding
        x = self.pos_embedding(x)
        x = self.layernorm_embed(x) # Apply LayerNorm after positional embedding
        x = self.input_dropout(x)

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)

        # Global Average Pooling (masked)
        if src_key_padding_mask is not None:
            # Invert mask: True for valid, False for padding
            mask = ~src_key_padding_mask.unsqueeze(-1).expand_as(x) # (batch, seq_len, embed_dim)
            # Sum valid elements along the sequence dimension
            masked_sum = torch.sum(x * mask, dim=1) # (batch, embed_dim)
            # Count valid elements along the sequence dimension (avoid division by zero)
            valid_count = mask.sum(dim=1) # (batch, embed_dim)
            # Clamp count to avoid division by zero if a sequence has no valid elements (shouldn't happen with proper padding)
            valid_count = torch.clamp(valid_count, min=1.0)
            pooled_output = masked_sum / valid_count # (batch, embed_dim)
        else:
            # If no mask, perform standard mean pooling
            pooled_output = torch.mean(x, dim=1) # (batch, embed_dim)


        # Pass through MLP head
        x = self.mlp_head(pooled_output)

        # Final output layer (logits)
        logits = self.output_layer(x) # (batch, n_classes)
        return logits
