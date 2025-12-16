import numpy as np
from collections import Counter
import torch

def rouge1_score(predictions, references):
    """
    Вычисление ROUGE-1 
    
    Args:
        predictions: список предсказанных токуенов (list of lists)
        references: список ожидаемых токенов (list of lists)
    
    Returns:
        precision, recall, f1_score
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for pred_tokens, ref_tokens in zip(predictions, references):
        if not ref_tokens or not pred_tokens:
            continue
            
        # Convert to sets for unigram comparison
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        
        # Calculate overlap
        overlap = len(pred_set.intersection(ref_set))
        
        # Calculate precision, recall, F1
        precision = overlap / len(pred_set) if len(pred_set) > 0 else 0
        recall = overlap / len(ref_set) if len(ref_set) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Return averages
    return {
        'precision': np.mean(precisions) if precisions else 0,
        'recall': np.mean(recalls) if recalls else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0
    }

def calculate_perplexity(loss):
    """вычисление экспоненты потерь"""
    return np.exp(loss)

def tokens_to_text(tokens, idx2word):
    """Преобразование индексов токенов"""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    texts = []
    for token_seq in tokens:
        words = []
        for idx in token_seq:
            if idx == 0:  # PAD token
                break
            if idx in idx2word:
                words.append(idx2word[idx])
            else:
                words.append('<UNK>')
        texts.append(' '.join(words))
    return texts
