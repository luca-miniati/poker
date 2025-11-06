import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class HandHistoryAutoencoder(nn.Module):
    '''LSTM-based autoencoder for variable-length hand history sequences.'''
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        '''
        Args:
            input_dim: Dimension of each action vector (hand_encoding_length from HandEncoder)
            hidden_dim: Hidden dimension of LSTM layers
            latent_dim: Dimension of compressed latent vector
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder: process sequence and compress to latent vector
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Compress bidirectional hidden state to latent vector
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: expand latent vector and reconstruct sequence
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        '''
        Encode variable-length sequences into fixed-size latent vectors.
        
        Args:
            x: (batch_size, max_seq_len, input_dim) padded sequences
            lengths: (batch_size,) original sequence lengths
        
        Returns:
            z: (batch_size, latent_dim) latent vectors
        '''
        # Pack padded sequences for efficient processing
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Run through encoder LSTM
        _, (h_n, c_n) = self.encoder_lstm(packed)
        
        # h_n shape: (num_layers * 2, batch_size, hidden_dim) for bidirectional
        # Reshape to (batch_size, num_layers * 2 * hidden_dim)
        h_n = h_n.transpose(0, 1).contiguous()
        h_n = h_n.view(h_n.size(0), -1)
        
        # Compress to latent vector
        z = self.encoder_fc(h_n)
        
        return z
    
    def decode(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        '''
        Decode latent vectors back into sequences.
        
        Args:
            z: (batch_size, latent_dim) latent vectors
            target_length: Length of sequences to generate
        
        Returns:
            x_recon: (batch_size, target_length, input_dim) reconstructed sequences
        '''
        batch_size = z.size(0)
        
        # Expand latent vector
        h = self.decoder_fc(z)  # (batch_size, hidden_dim)
        
        # Repeat for each timestep
        h = h.unsqueeze(1).repeat(1, target_length, 1)  # (batch_size, target_length, hidden_dim)
        
        # Run through decoder LSTM
        lstm_out, _ = self.decoder_lstm(h)
        
        # Project to output dimension
        x_recon = self.output_fc(lstm_out)
        
        return x_recon
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        target_length: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Full autoencoder forward pass.
        
        Args:
            x: (batch_size, max_seq_len, input_dim) input sequences
            lengths: (batch_size,) sequence lengths
            target_length: Length to decode to (defaults to max length in batch)
        
        Returns:
            x_recon: (batch_size, target_length, input_dim) reconstructed sequences
            z: (batch_size, latent_dim) latent vectors
        '''
        if target_length is None:
            target_length = x.size(1)
        
        # Encode
        z = self.encode(x, lengths)
        
        # Decode
        x_recon = self.decode(z, target_length)
        
        return x_recon, z


class AutoencoderLoss(nn.Module):
    '''Masked reconstruction loss for variable-length sequences.'''
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute masked MSE loss.
        
        Args:
            x_recon: (batch_size, seq_len, input_dim) reconstructed sequences
            x_target: (batch_size, seq_len, input_dim) target sequences
            mask: (batch_size, seq_len) boolean mask (1 for real data, 0 for padding)
        
        Returns:
            loss: scalar loss value
        '''
        # Compute element-wise MSE
        loss = self.mse(x_recon, x_target)  # (batch_size, seq_len, input_dim)
        
        # Sum over feature dimension
        loss = loss.sum(dim=-1)  # (batch_size, seq_len)
        
        # Apply mask and normalize
        mask = mask.float()
        loss = loss * mask
        
        if self.reduction == 'mean':
            # Average over non-padded elements
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def count_parameters(model: nn.Module) -> int:
    '''Count trainable parameters in model.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)