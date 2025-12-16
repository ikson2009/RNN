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
        
        # конвертация в нижний регситр
        text = text.lower()
        
        # убираем URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # убираем хэштаги
        text = re.sub(r'@\w+|\#', '', text)
        
        # отсавляем только буквы , числа и пункуацию
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # убираем пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # преобразуемв токены и считаем частоты
        word_counts = Counter()
        for text in texts:
            tokens = text.split()
            word_counts.update(tokens)
        
        # фильтруем по частоте
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_freq]
        
        # ограничиваем размер словаря 
        if len(filtered_words) > self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]
        
        # дополняем словарь спец токенами
        self.vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + filtered_words
        self.vocab_size = len(self.vocab)
        
        # Отображение слова-индексы словаря и обратно
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        return self.vocab
    
    def text_to_sequences(self, texts, max_length=None):
        """преобразование текста в последовательность индекстов словаря """
        if max_length is None:
            max_length = self.seq_length
            
        sequences = []
        for text in texts:
            tokens = text.split()
            # Add SOS and EOS tokens
            tokens = ['<SOS>'] + tokens + ['<EOS>']
            # Convert to indices
            seq = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                  for token in tokens]
            # Pad or truncate
            if len(seq) < max_length:
                seq = seq + [self.word2idx['<PAD>']] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def sequences_to_text(self, sequences):
        """Псоледовательность индексов словаря в текст """
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
        """Подготовка данных для задачи autocomplete"""
        if predict_n_tokens is None:
            predict_n_tokens = self.predict_n_tokens
            
        X = []
        y = []
        
        pad_idx = self.word2idx['<PAD>']
        
        for seq in sequences:
            # преобразование в список 
            seq_list = seq.tolist() if isinstance(seq, np.ndarray) else list(seq)
            
            # Нахождение токенов (before padding)
            actual_tokens = []
            for token in seq_list:
                if token == pad_idx:
                    break
                actual_tokens.append(token)
            
            seq_len = len(actual_tokens)
            
            # Если короткий - пропускаем
            if seq_len <= predict_n_tokens + 1:
                continue
            
            # Создаем тренировочные примеры
            for i in range(1, seq_len - predict_n_tokens):
                # Input: tokens before position i
                input_tokens = actual_tokens[:i]
                
                # Ensure input has exactly seq_length tokens
                if len(input_tokens) > self.seq_length:
                    input_tokens = input_tokens[-self.seq_length:]  # Keep most recent
                elif len(input_tokens) < self.seq_length:
                    # Pad at the end
                    input_tokens =  [pad_idx] * (self.seq_length - len(input_tokens)) + input_tokens 
                
                # Target: next predict_n_tokens tokens
                target_tokens = actual_tokens[i:i+predict_n_tokens]
                
                # Ensure target has exactly predict_n_tokens
                if len(target_tokens) < predict_n_tokens:
                    target_tokens = target_tokens + [pad_idx] * (predict_n_tokens - len(target_tokens))
                
                # Add to lists
                X.append(input_tokens)
                y.append(target_tokens)
        
        # Convert to numpy arrays
        if X and y:
            X_array = np.array(X, dtype=np.int64)
            y_array = np.array(y, dtype=np.int64)
            # print(f"X={X}")
            # print(f"y={y}")
            return X_array, y_array
        else:
            return np.array([]), np.array([])
    
    def save(self, filepath):
        """Сохранение состояния препроцессора """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'config': self.config
            }, f)
    
    def load(self, filepath):
        """Загрузка состояния препроцессора """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            if 'config' in data:
                self.config = data['config']

def load_sentiment140_data(config):
    """Загрузка и предобработка sentiment140"""
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
        
        # Build vocabulary from training data
        preprocessor.build_vocab(train_df['cleaned_text'].tolist())
        
        # Convert texts to sequences
        train_sequences = preprocessor.text_to_sequences(train_df['cleaned_text'].tolist())
        val_sequences = preprocessor.text_to_sequences(val_df['cleaned_text'].tolist())
        
        # Prepare autocomplete data
        X_train, y_train = preprocessor.prepare_autocomplete_data(train_sequences)
        X_val, y_val = preprocessor.prepare_autocomplete_data(val_sequences)
        
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
