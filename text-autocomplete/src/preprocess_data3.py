import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import pickle

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.max_vocab_size = config.max_vocab_size
        self.min_freq = config.min_freq
        self.seq_length = config.seq_length
        self.predict_n_tokens = config.predict_n_tokens
        self.vocab = None
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#', '', text)
        
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Tokenize and count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = text.split()
            word_counts.update(tokens)
        
        # Filter by minimum frequency
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_freq]
        
        # Limit vocabulary size
        if len(filtered_words) > self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]
        
        # Create vocabulary with special tokens
        self.vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + filtered_words
        self.vocab_size = len(self.vocab)
        
        # Create mapping dictionaries
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        print(self.vocab)
        return self.vocab
    
    def text_to_sequences(self, texts, max_length=None):
        """Convert texts to sequences of indices"""
        if max_length is None:
            # max_length = self.seq_length
            max_length = 140
            
        sequences = []
        for text in texts:
            tokens = text.split()
            # Add SOS and EOS tokens
            tokens = ['<SOS>'] + tokens + ['<EOS>']
            print(f"tokens + {tokens}")
            # Convert to indices
            seq = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                  for token in tokens]
            # Pad or truncate
            if len(seq) < max_length:
                seq = [self.word2idx['<PAD>']] * (max_length - len(seq)) + seq 
            else:
                seq = seq[:max_length]

            sequences.append(seq)
            print(f"seq= {seq}" )
        print(f"sequences= {sequences}")
        return np.array(sequences)
    
    def sequences_to_text(self, sequences):
        """Convert sequences of indices back to text"""
        texts = []
        for seq in sequences:
            tokens = []
            for idx in seq:
                if idx == self.word2idx['<PAD>']:
                    break
                if idx == self.word2idx['<EOS>']:
                    break
                if idx == self.word2idx['<SOS>']:
                    continue
                tokens.append(self.idx2word.get(idx, '<UNK>'))
            texts.append(' '.join(tokens))
        return texts
    
    def prepare_autocomplete_data(self, sequences, predict_n_tokens=None):
        """Prepare data for autocomplete task"""
        if predict_n_tokens is None:
            predict_n_tokens = self.predict_n_tokens
            
        X = []
        y = []
        
        for seq in sequences:
            # Find actual length of sequence (excluding padding)
            actual_tokens = []
            for token in seq:
                if token == self.word2idx['<PAD>']:
                    break
                actual_tokens.append(token)
            
            seq_len = len(actual_tokens)
            
            if seq_len <= predict_n_tokens + 1:
                continue
                
            for i in range(1, seq_len - predict_n_tokens):
                # Input: everything up to position i
                input_seq = actual_tokens[:i]
                
                # Pad input sequence to seq_length
                if len(input_seq) < self.seq_length:
                    # Pad at the end
                    padded_input = input_seq + [self.word2idx['<PAD>']] * (self.seq_length - len(input_seq))
                else:
                    # Truncate from the beginning to keep most recent tokens
                    padded_input = input_seq[-self.seq_length:]
                
                # Target: next predict_n_tokens tokens
                target_seq = actual_tokens[i:i+predict_n_tokens]
                
                # Pad target sequence if needed (should always be predict_n_tokens)
                if len(target_seq) < predict_n_tokens:
                    target_seq = target_seq + [self.word2idx['<PAD>']] * (predict_n_tokens - len(target_seq))
                
                X.append(padded_input)
                y.append(target_seq)
        print(f'prepare_autocomplete_data input {X}')
        print(f'prepare_autocomplete_data input {y}')
        
        return np.array(X), np.array(y)
    
    def save(self, filepath):
        """Save preprocessor state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'config': self.config
            }, f)
    
    def load(self, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            if 'config' in data:
                self.config = data['config']

def load_sentiment140_data(config):
    """Load and prepare sentiment140 dataset"""
    try:
        # Load dataset
        df = pd.read_csv(config.data_path, encoding='latin-1', header=None)
        df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
        
        # Clean texts
        preprocessor = TextPreprocessor(config)
        df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
        
        # Filter out empty texts
        df = df[df['cleaned_text'].str.strip() != '']
        
        # Split into train and validation
        train_df, val_df = train_test_split(
            df, 
            test_size=config.test_size, 
            random_state=config.random_seed
        )
        # print(f"train_df is = {train_df['cleaned_text'].tolist()}")
        
        # Build vocabulary from training data
        # preprocessor.build_vocab(train_df['cleaned_text'].tolist())
        preprocessor.build_vocab(df['cleaned_text'].tolist())
        
        # Convert texts to sequences
        train_sequences = preprocessor.text_to_sequences(train_df['cleaned_text'].tolist())
        val_sequences = preprocessor.text_to_sequences(val_df['cleaned_text'].tolist())
        
        # Prepare autocomplete data
        X_train, y_train = preprocessor.prepare_autocomplete_data(train_sequences)
        X_val, y_val = preprocessor.prepare_autocomplete_data(val_sequences)
        print(X_train)
        print(y_train)
        
        return {
            'preprocessor': preprocessor,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'train_texts': train_df['cleaned_text'].tolist(),
            'val_texts': val_df['cleaned_text'].tolist()
        }
    
    except FileNotFoundError:
        print(f"Dataset not found at {config.data_path}")
        print("Download from: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")
        return None