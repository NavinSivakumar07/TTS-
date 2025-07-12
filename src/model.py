import torch
import torch.nn as nn
from src import config

class Encoder(nn.Module):
    """
    A simple encoder module.
    It consists of an embedding layer followed by a few convolutional layers.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.N_SYMBOLS, config.EMBEDDING_DIM)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.EMBEDDING_DIM, config.HIDDEN_DIM, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Conv1d(config.HIDDEN_DIM, config.HIDDEN_DIM, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2) # (batch_size, embedding_dim, seq_len)
        x = self.conv_layers(x) # (batch_size, hidden_dim, seq_len)
        return x

class Decoder(nn.Module):
    """
    A simple decoder module.
    It takes the encoded representation and outputs a mel-spectrogram.
    For this toy example, we'll use a linear layer to predict the fixed-size mel-spectrogram.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # We need to figure out the input size to the linear layer.
        # Let's assume the encoder output is averaged over the time dimension.
        self.linear = nn.Linear(
            config.HIDDEN_DIM, 
            config.MEL_CHANNELS * config.MEL_SPEC_LENGTH
        )
        
    def forward(self, x):
        # x: (batch_size, hidden_dim, seq_len)
        # Average pooling over the sequence length
        x = x.mean(dim=2) # (batch_size, hidden_dim)
        x = self.linear(x) # (batch_size, n_mels * mel_length)
        
        # Reshape to (batch_size, n_mels, mel_length)
        x = x.view(
            -1, 
            config.MEL_CHANNELS, 
            config.MEL_SPEC_LENGTH
        )
        return x

class SimpleTTS(nn.Module):
    """
    A simplified Text-to-Speech model.
    """
    def __init__(self):
        super(SimpleTTS, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, text_seq):
        encoded = self.encoder(text_seq)
        mel_output = self.decoder(encoded)
        return mel_output

if __name__ == '__main__':
    # A simple test to verify the model architecture
    model = SimpleTTS()
    
    # Create a dummy input batch
    dummy_input = torch.randint(0, config.N_SYMBOLS, (4, 50)) # (batch_size, seq_len)
    
    # Forward pass
    output = model(dummy_input)
    
    print("Model Test")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Expected output shape:", (4, config.MEL_CHANNELS, config.MEL_SPEC_LENGTH))
    assert output.shape == (4, config.MEL_CHANNELS, config.MEL_SPEC_LENGTH)
    print("\\nModel architecture test passed.")
