import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import duckdb
import numpy as np
from typing import List, Tuple
from data.model.hand import Hand
from data.utils.hand_encoder import HandEncoder


class HandHistoryDataset(Dataset):
    '''Dataset for loading and encoding poker hand histories.'''
    
    def __init__(self, db_path: str, hand_ids: List[str] = None, max_actions: int = 200):
        '''
        Args:
            db_path: Path to DuckDB database
            hand_ids: List of hand IDs to load. If None, loads all hands.
            max_actions: Maximum number of actions to keep per hand (truncate longer hands)
        '''
        self.db_path = db_path
        self.max_actions = max_actions
        self.encoder = HandEncoder()
        
        # Connect to database and get hand IDs
        self.con = duckdb.connect(db_path, read_only=True)
        
        if hand_ids is None:
            # Load all hand IDs from database
            result = self.con.sql('SELECT hand_id FROM hands').df()
            self.hand_ids = result['hand_id'].tolist()
        else:
            self.hand_ids = hand_ids
        
        print(f'Loaded {len(self.hand_ids)} hands from database')
    
    def __len__(self) -> int:
        return len(self.hand_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        '''
        Returns:
            encoded_hand: (seq_len, 119) tensor
            original_length: actual number of actions before padding
        '''
        hand_id = self.hand_ids[idx]
        
        # Load hand from database
        hand = Hand.from_db(hand_id, self.con)
        
        # Encode using your existing encoder
        encoded = self.encoder.encode(hand)  # (num_actions, 119)
        
        # Truncate if too long
        if len(encoded) > self.max_actions:
            encoded = encoded[:self.max_actions]
        
        original_length = len(encoded)
        
        # Convert to tensor
        tensor = torch.from_numpy(encoded).float()
        
        return tensor, original_length
    
    def close(self):
        '''Close database connection.'''
        self.con.close()


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of (tensor, length) tuples
    
    Returns:
        padded_sequences: (batch_size, max_seq_len, 119) padded tensor
        lengths: (batch_size,) original sequence lengths
        mask: (batch_size, max_seq_len) boolean mask (1 for real data, 0 for padding)
    '''
    sequences, lengths = zip(*batch)
    
    # Pad sequences to same length
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Create mask: 1 for real data, 0 for padding
    batch_size, max_len, _ = padded.shape
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded, lengths_tensor, mask


def create_train_val_dataloaders(
    db_path: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    max_actions: int = 200,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    '''
    Create train and validation dataloaders with automatic train/val split.
    
    Args:
        db_path: Path to DuckDB database
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        max_actions: Maximum actions per hand
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    '''
    # Load all hand IDs
    con = duckdb.connect(db_path, read_only=True)
    all_hand_ids = con.sql('SELECT hand_id FROM hands').df()['hand_id'].tolist()
    con.close()
    
    # Shuffle and split
    np.random.shuffle(all_hand_ids)
    split_idx = int(len(all_hand_ids) * (1 - val_split))
    train_ids = all_hand_ids[:split_idx]
    val_ids = all_hand_ids[split_idx:]
    
    print(f'Split: {len(train_ids)} train, {len(val_ids)} validation')
    
    # Create datasets
    train_dataset = HandHistoryDataset(db_path, train_ids, max_actions)
    val_dataset = HandHistoryDataset(db_path, val_ids, max_actions)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader