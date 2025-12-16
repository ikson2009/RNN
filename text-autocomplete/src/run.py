#!/usr/bin/env python3
"""
Command-line script to run the autocomplete system
"""

import sys
import os
from config_loader import Config

def main():
    """Main function to run the autocomplete system"""
    
    # Load configuration
    config_path = 'config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = Config(config_path)
    
    print("="*60)
    print("Autocomplete System: LSTM vs DistilGPT2")
    print("="*60)
    
    # Run the main notebook logic here or import it
    # For simplicity, we'll just show the configuration
    print("\nConfiguration loaded successfully:")
    print(f"  Data path: {config.data_path}")
    print(f"  Model save path: {config.model_save}")
    print(f"  Number of epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    
    print("\nTo run the full experiment, execute the main.ipynb notebook")
    print("or import and run the modules individually.")

if __name__ == "__main__":
    main()