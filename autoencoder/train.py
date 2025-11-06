import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from .dataset import create_train_val_dataloaders
from .model import HandHistoryAutoencoder, AutoencoderLoss, count_parameters


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    '''Train for one epoch.'''
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for _, (x, lengths, mask) in enumerate(pbar):
        x = x.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        
        # Forward pass
        x_recon, z = model(x, lengths)
        loss = criterion(x_recon, x, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, loader, criterion, device, epoch):
    '''Validate the model.'''
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for x, lengths, mask in pbar:
            x = x.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            
            # Forward pass
            x_recon, z = model(x, lengths)
            loss = criterion(x_recon, x, mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train hand history autoencoder')
    
    # Data
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--max_actions', type=int, default=200,
                        help='Maximum actions per hand')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of LSTM')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='runs/autoencoder',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader = create_train_val_dataloaders(
        db_path=args.db_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_actions=args.max_actions,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print('\nInitializing model...')
    model = HandHistoryAutoencoder(
        input_dim=119,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    print(f'Model has {count_parameters(model):,} trainable parameters')
    print(f'Latent dimension: {args.latent_dim}')
    
    # Loss and optimizer
    criterion = AutoencoderLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,
        factor=0.5,
    )
    
    # Training loop
    print('\nStarting training...')
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.checkpoint_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f'Saved best model with val loss {val_loss:.4f}')
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
    
    writer.close()
    print('\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()