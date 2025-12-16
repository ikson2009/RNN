import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
from eval_metrics import rouge1_score, tokens_to_text

def train_lstm(model, train_loader, val_loader, preprocessor, config):
    
    # Move model to device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.factor, 
        patience=config.patience
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rouge': [],
        'val_rouge': [],
        'learning_rate': []
    }
    
    print(f"Training on {device}")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_references = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, _ = model(inputs)
            
            # Calculate loss for each predicted token
            loss = 0
            batch_size = targets.size(0)
            for i in range(model.predict_n_tokens):
                # Reshape predictions for cross entropy
                pred_i = predictions[:, i, :].contiguous().view(-1, predictions.size(-1))
                target_i = targets[:, i].contiguous().view(-1)
                loss += criterion(pred_i, target_i)
            
            # Average loss
            loss = loss / model.predict_n_tokens
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions and references for ROUGE calculation
            pred_tokens = torch.argmax(predictions, dim=-1).cpu().numpy()
            target_tokens = targets.cpu().numpy()
            
            # Convert to text for ROUGE
            pred_texts = tokens_to_text(pred_tokens, preprocessor.idx2word)
            ref_texts = tokens_to_text(target_tokens, preprocessor.idx2word)
            
            train_predictions.extend([text.split() for text in pred_texts])
            train_references.extend([text.split() for text in ref_texts])
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_rouge = rouge1_score(train_predictions, train_references)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_references = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Val]'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                predictions, _ = model(inputs)
                
                # Calculate validation loss
                loss = 0
                for i in range(model.predict_n_tokens):
                    pred_i = predictions[:, i, :].contiguous().view(-1, predictions.size(-1))
                    target_i = targets[:, i].contiguous().view(-1)
                    loss += criterion(pred_i, target_i)
                loss = loss / model.predict_n_tokens
                
                val_loss += loss.item()
                
                # Store predictions and references
                pred_tokens = torch.argmax(predictions, dim=-1).cpu().numpy()
                target_tokens = targets.cpu().numpy()
                
                pred_texts = tokens_to_text(pred_tokens, preprocessor.idx2word)
                ref_texts = tokens_to_text(target_tokens, preprocessor.idx2word)
                
                val_predictions.extend([text.split() for text in pred_texts])
                val_references.extend([text.split() for text in ref_texts])
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_rouge = rouge1_score(val_predictions, val_references)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rouge'].append(train_rouge)
        history['val_rouge'].append(val_rouge)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train ROUGE-1 F1: {train_rouge['f1']:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val ROUGE-1 F1: {val_rouge['f1']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Print some examples
        print(f"\n  Examples (first {min(config.num_examples_display, len(val_predictions))}):")
        for i in range(min(config.num_examples_display, len(val_predictions))):
            print(f"    Input: ... (context)")
            print(f"    Predicted: {' '.join(val_predictions[i])}")
            print(f"    Actual: {' '.join(val_references[i])}")
            print()
    
    return model, history

def save_model(model, preprocessor, filepath):
    """Save model and preprocessor"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'predict_n_tokens': model.predict_n_tokens,
            'bidirectional': model.bidirectional,
            'config': model.config
        },
        'preprocessor': preprocessor
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, config, device='cuda'):
    """Load model and preprocessor"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model configuration
    model_config = config.get_model_config()
    model_config['vocab_size'] = checkpoint['model_config']['vocab_size']
    
    # Create model
    model = LSTMAutocomplete(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get preprocessor
    preprocessor = checkpoint['preprocessor']
    preprocessor.config = config  # Update with current config
    
    return model, preprocessor