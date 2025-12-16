import yaml
import os

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Загрузка YAML файла"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Data parameters
        self.data_path = config['data']['data_path']
        self.max_vocab_size = config['data']['max_vocab_size']
        self.min_freq = config['data']['min_freq']
        self.seq_length = config['data']['seq_length']
        self.predict_n_tokens = config['data']['predict_n_tokens']
        self.test_size = config['data']['test_size']
        self.random_seed = config['data']['random_seed']
        self.num_samples_gpt2_eval = config['data']['num_samples_gpt2_eval']
        self.batch_size = config['data']['batch_size']
        
        # Model parameters
        self.embedding_dim = config['model']['embedding_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.bidirectional = config['model']['bidirectional']
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.patience = config['training']['patience']
        self.factor = config['training']['factor']
        self.clip_grad_norm = config['training']['clip_grad_norm']
        self.device = config['training']['device']
        
        # Evaluation parameters
        self.num_examples_display = config['evaluation']['num_examples_display']
        self.num_differing_examples = config['evaluation']['num_differing_examples']
        self.interactive_test_samples = config['evaluation']['interactive_test_samples']
        
        # GPT-2 parameters
        self.gpt2_model_name = config['gpt2']['model_name']
        self.max_length_padding = config['gpt2']['max_length_padding']
        self.do_sample = config['gpt2']['do_sample']
        self.num_return_sequences = config['gpt2']['num_return_sequences']
        
        # File paths
        self.model_save = config['paths']['model_save']
        self.preprocessor_save = config['paths']['preprocessor_save']
        self.comparison_csv = config['paths']['comparison_csv']
        self.comparison_plot = config['paths']['comparison_plot']
        self.training_plot = config['paths']['training_plot']
        self.final_results = config['paths']['final_results']
    
    def get_model_config(self):
        """Конфигунация модели"""
        return {
            'vocab_size': None,  # To be filled later
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seq_length': self.seq_length,
            'predict_n_tokens': self.predict_n_tokens,
            'bidirectional': self.bidirectional
        }
    
    def get_data_config(self):
        """Конфигурация данных"""
        return {
            'data_path': self.data_path,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq,
            'seq_length': self.seq_length,
            'predict_n_tokens': self.predict_n_tokens,
            'test_size': self.test_size,
            'random_seed': self.random_seed
        }
    
    def get_training_config(self):
        """Конфигурация обучения"""
        return {
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'factor': self.factor,
            'clip_grad_norm': self.clip_grad_norm,
            'device': self.device
        }
