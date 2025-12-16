import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutocomplete(nn.Module):
    def __init__(self, config):
        super(LSTMAutocomplete, self).__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.predict_n_tokens = config.predict_n_tokens
        self.bidirectional = config.bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim, 
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # Adjust hidden dimension for bidirectional
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # Output layers for multiple tokens
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(lstm_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.vocab_size)
            ) for _ in range(self.predict_n_tokens)
        ])
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout_layer(embedded)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        # lstm_out shape: (batch_size, seq_length, hidden_dim * num_directions)
        
        # Get the last hidden state for each sequence
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        # Generate predictions for multiple tokens
        predictions = []
        for i in range(self.predict_n_tokens):
            pred = self.output_layers[i](last_hidden)  # (batch_size, vocab_size)
            predictions.append(pred)
        
        # Stack predictions
        # Shape: (batch_size, predict_n_tokens, vocab_size)
        predictions = torch.stack(predictions, dim=1)
        
        return predictions, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        num_directions = 2 if self.bidirectional else 1
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(device),
                weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(device))