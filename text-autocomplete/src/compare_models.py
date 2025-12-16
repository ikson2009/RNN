import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compare_models(lstm_results, gpt2_results, config, save_path=None):
    """сравнение LSTM и distilgpt2"""
    if save_path is None:
        save_path = config.comparison_plot
    
    # Create comparison dataframe
    comparison_data = {
        'Model': ['LSTM', 'DistilGPT2'],
        'ROUGE-1 Precision': [
            lstm_results['rouge1']['precision'],
            gpt2_results['rouge1']['precision']
        ],
        'ROUGE-1 Recall': [
            lstm_results['rouge1']['recall'],
            gpt2_results['rouge1']['recall']
        ],
        'ROUGE-1 F1': [
            lstm_results['rouge1']['f1'],
            gpt2_results['rouge1']['f1']
        ]
    }
    
    if 'perplexity' in lstm_results:
        comparison_data['Perplexity'] = [
            lstm_results['perplexity'],
            'N/A'  # GPT-2 doesn't provide loss for this task
        ]
    
    df = pd.DataFrame(comparison_data)
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("\n" + "="*60)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Precision, Recall, F1 Comparison
    metrics = ['ROUGE-1 Precision', 'ROUGE-1 Recall', 'ROUGE-1 F1']
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        lstm_val = df[df['Model'] == 'LSTM'][metric].values[0]
        gpt2_val = df[df['Model'] == 'DistilGPT2'][metric].values[0]
        
        axes[i].bar(['LSTM', 'DistilGPT2'], [lstm_val, gpt2_val], color=['blue', 'orange'])
        axes[i].set_title(metric)
        axes[i].set_ylabel('Score')
        axes[i].set_ylim([0, 1])
        
        # Add value labels
        for j, v in enumerate([lstm_val, gpt2_val]):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    # Calculate percentage difference
    print("\nPERFORMANCE DIFFERENCE (GPT2 - LSTM):")
    print("-"*40)
    for metric in metrics:
        lstm_val = df[df['Model'] == 'LSTM'][metric].values[0]
        gpt2_val = df[df['Model'] == 'DistilGPT2'][metric].values[0]
        diff = gpt2_val - lstm_val
        diff_pct = (diff / lstm_val) * 100 if lstm_val > 0 else 0
        
        print(f"{metric}:")
        print(f"  LSTM: {lstm_val:.4f}")
        print(f"  GPT-2: {gpt2_val:.4f}")
        print(f"  Difference: {diff:+.4f} ({diff_pct:+.1f}%)")
        print()
    
    # Determine which model performed better
    lstm_f1 = df[df['Model'] == 'LSTM']['ROUGE-1 F1'].values[0]
    gpt2_f1 = df[df['Model'] == 'DistilGPT2']['ROUGE-1 F1'].values[0]
    
    if gpt2_f1 > lstm_f1:
        print(f"DistilGPT2 outperformed LSTM by {(gpt2_f1 - lstm_f1):.4f} in F1 score")
    elif lstm_f1 > gpt2_f1:
        print(f"LSTM outperformed DistilGPT2 by {(lstm_f1 - gpt2_f1):.4f} in F1 score")
    else:
        print("Both models performed similarly")
    
    return df

def analyze_errors(lstm_predictions, gpt2_predictions, references, config):
    """Анализ ошибок при предсказании"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Find examples where models differ
    differing_examples = []
    for i, (lstm_pred, gpt2_pred, ref) in enumerate(zip(lstm_predictions, gpt2_predictions, references)):
        lstm_text = ' '.join(lstm_pred) if isinstance(lstm_pred, list) else lstm_pred
        gpt2_text = ' '.join(gpt2_pred) if isinstance(gpt2_pred, list) else gpt2_pred
        ref_text = ' '.join(ref) if isinstance(ref, list) else ref
        
        if lstm_text != gpt2_text:
            differing_examples.append((i, lstm_text, gpt2_text, ref_text))
    
    print(f"\nFound {len(differing_examples)} examples where predictions differ")
    
    # Show some examples
    num_examples = min(config.num_differing_examples, len(differing_examples))
    print(f"\nFirst {num_examples} differing examples:")
    print("-"*60)
    
    for i, (idx, lstm_pred, gpt2_pred, ref) in enumerate(differing_examples[:num_examples]):
        print(f"\nExample {i+1} (Index {idx}):")
        print(f"  LSTM Prediction: {lstm_pred}")
        print(f"  GPT-2 Prediction: {gpt2_pred}")
        print(f"  Actual: {ref}")
        
        # Check which is closer
        lstm_match = lstm_pred == ref
        gpt2_match = gpt2_pred == ref
        
        if lstm_match and not gpt2_match:
            print("  ✓ LSTM is correct")
        elif gpt2_match and not lstm_match:
            print("  ✓ GPT-2 is correct")
        elif lstm_match and gpt2_match:
            print("  ✓ Both are correct")
        else:
            print("  ✗ Both are incorrect")
