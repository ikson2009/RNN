from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm
from eval_metrics import rouge1_score

class DistilGPT2Autocomplete:
    def __init__(self, config):
        self.config = config
        self.model_name = config.gpt2_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.generator = pipeline('text-generation', model=self.model_name, tokenizer=self.tokenizer)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def predict_next_tokens(self, input_text, num_tokens=None, num_return_sequences=None):
        """Предсказание токена используя distilgpt2"""
        if num_tokens is None:
            num_tokens = self.config.predict_n_tokens
        if num_return_sequences is None:
            num_return_sequences = self.config.num_return_sequences
            
        try:
            # Generate text
            outputs = self.generator(
                input_text,
                max_length=len(self.tokenizer.encode(input_text)) + num_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=self.config.do_sample
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Get only the new tokens
            input_tokens = self.tokenizer.encode(input_text)
            all_tokens = self.tokenizer.encode(generated_text)
            new_tokens = all_tokens[len(input_tokens):]
            
            # Decode new tokens
            predicted_tokens = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split()
            
            # Return only requested number of tokens
            return predicted_tokens[:num_tokens]
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return []
    
    def evaluate_on_dataset(self, texts, references, num_tokens=None, batch_size=None):
        """оценка distilgpt2 по датасету"""
        if num_tokens is None:
            num_tokens = self.config.predict_n_tokens
        if batch_size is None:
            batch_size = self.config.batch_size
            
        all_predictions = []
        all_references = []
        
        print(f"Evaluating distilgpt2 on {len(texts)} samples...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_refs = references[i:i+batch_size]
            
            for text, ref in zip(batch_texts, batch_refs):
                # Prepare input (last few words as context)
                words = text.split()
                if len(words) > self.config.max_length_padding:
                    context = ' '.join(words[-self.config.max_length_padding:])
                else:
                    context = text
                
                # Predict next tokens
                predicted_tokens = self.predict_next_tokens(context, num_tokens=num_tokens)
                
                # Store results
                all_predictions.append(predicted_tokens)
                all_references.append(ref.split()[:num_tokens])
        
        # Calculate ROUGE scores
        rouge_scores = rouge1_score(all_predictions, all_references)
        
        print("\n" + "="*50)
        print("DISTILGPT2 EVALUATION")
        print("="*50)
        print(f"ROUGE-1 Precision: {rouge_scores['precision']:.4f}")
        print(f"ROUGE-1 Recall: {rouge_scores['recall']:.4f}")
        print(f"ROUGE-1 F1: {rouge_scores['f1']:.4f}")
        
        # Print some examples
        print(f"\nExamples (first {min(self.config.num_examples_display, len(all_predictions))}):")
        print("-"*50)
        for i in range(min(self.config.num_examples_display, len(all_predictions))):
            print(f"Example {i+1}:")
            words = texts[i].split()
            if len(words) > 5:
                context_words = words[-5:]
            else:
                context_words = words
            print(f"  Context: ... {' '.join(context_words)}")
            print(f"  Predicted: {' '.join(all_predictions[i])}")
            print(f"  Actual: {' '.join(all_references[i])}")
            print()
        
        return {
            'rouge1': rouge_scores,
            'predictions': all_predictions,
            'references': all_references
        }

def create_distilgpt2_test_data(preprocessor, texts, config):
    """Создание тестовых данных для оценки distilgpt2"""
    test_texts = []
    test_references = []
    
    for text in texts[:config.num_samples_gpt2_eval]:
        words = text.split()
        if len(words) > config.predict_n_tokens + 1:
            # Use first n-predict_n_tokens words as context
            context_words = words[:-(config.predict_n_tokens)]
            # Last predict_n_tokens words as reference
            ref_words = words[-(config.predict_n_tokens):]
            
            test_texts.append(' '.join(context_words))
            test_references.append(' '.join(ref_words))
    
    return test_texts, test_references
