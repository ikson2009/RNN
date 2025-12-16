import torch
import numpy as np
from tqdm import tqdm
from eval_metrics import rouge1_score, tokens_to_text

def evaluate_lstm(model, dataloader, preprocessor, config):
    """оценка LSTM модели"""
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_references = []
    all_losses = []
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating LSTM"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # print(f"inputs= {inputs}")
            predictions, _ = model(inputs)
            # print(f"predictions= {predictions}")
            
            # Calculate loss
            loss = 0
            for i in range(model.predict_n_tokens):
                pred_i = predictions[:, i, :].contiguous().view(-1, predictions.size(-1))
                target_i = targets[:, i].contiguous().view(-1)
                loss += criterion(pred_i, target_i)
            loss = loss / model.predict_n_tokens
            all_losses.append(loss.item())
            
            # Convert to tokens
            pred_tokens = torch.argmax(predictions, dim=-1).cpu().numpy()
            target_tokens = targets.cpu().numpy()
            
            # Convert to text
            pred_texts = tokens_to_text(pred_tokens, preprocessor.idx2word)
            ref_texts = tokens_to_text(target_tokens, preprocessor.idx2word)
            
            all_predictions.extend([text.split() for text in pred_texts])
            all_references.extend([text.split() for text in ref_texts])
    
    # Calculate metrics
    avg_loss = np.mean(all_losses)
    rouge_scores = rouge1_score(all_predictions, all_references)
    
    print("\n" + "="*50)
    print("LSTM MODEL EVALUATION")
    print("="*50)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {np.exp(avg_loss):.4f}")
    print(f"ROUGE-1 Precision: {rouge_scores['precision']:.4f}")
    print(f"ROUGE-1 Recall: {rouge_scores['recall']:.4f}")
    print(f"ROUGE-1 F1: {rouge_scores['f1']:.4f}")
    
    # Print some examples
    print(f"\nExamples (first {min(config.num_examples_display, len(all_predictions))}):")
    print("-"*50)
    for i in range(min(config.num_examples_display, len(all_predictions))):
        print(f"Example {i+1}:")
        print(f"  Predicted: {' '.join(all_predictions[i])}")
        print(f"  Actual: {' '.join(all_references[i])}")
        print()
    
    return {
        'loss': avg_loss,
        'perplexity': np.exp(avg_loss),
        'rouge1': rouge_scores,
        'predictions': all_predictions,
        'references': all_references
    }

def predict_next_tokens(model, preprocessor, input_text, config, num_predictions=None):
    """предсказание следующего токена по тексту"""
    if num_predictions is None:
        num_predictions = config.num_examples_display
        
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # очистка и токенизация входа 
    cleaned_text = preprocessor.clean_text(input_text)
    tokens = cleaned_text.split()[-preprocessor.seq_length+1:]  # Leave space for SOS
    
    # преобразование в последовательность
    seq = [preprocessor.word2idx['<SOS>']] + \
          [preprocessor.word2idx.get(token, preprocessor.word2idx['<UNK>']) 
           for token in tokens]
    
    # дополнение последовательности
    if len(seq) < preprocessor.seq_length:
        seq = seq + [preprocessor.word2idx['<PAD>']] * (preprocessor.seq_length - len(seq))
    else:
        seq = seq[:preprocessor.seq_length]
    
    # в тензор
    input_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions, _ = model(input_tensor)
        pred_tokens = torch.argmax(predictions, dim=-1).squeeze(0).cpu().numpy()
    
    # в текст
    predicted_words = []
    for token in pred_tokens:
        if token == preprocessor.word2idx['<PAD>'] or token == preprocessor.word2idx['<EOS>']:
            break
        predicted_words.append(preprocessor.idx2word.get(token, '<UNK>'))
    
    print(f"Input: {input_text}")
    print(f"Predicted next {model.predict_n_tokens} tokens: {' '.join(predicted_words)}")
    
    return predicted_words
