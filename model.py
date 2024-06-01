import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_encoding[:seq_len, :]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, num_classes, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc1=nn.Linear(d_model,256)
        self.fc2=nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_classes)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        #x = self.layer_norm(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
