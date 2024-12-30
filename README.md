# GPT2Trainer: Advanced Training Module for GPT-2

GPT2Trainer is a comprehensive training utility for GPT-2 models, offering advanced features like device selection, gradient accumulation, checkpoint management, and extensive logging to facilitate efficient and reliable training processes. The model trainer has features for optimized training efficiency, including automatic hardware detection prioritizing NVIDIA GPU with fallback options like Intel OpenVINO and CPU. It customizes tokenizers by adding padding tokens and resizing model embeddings. Training is enhanced with manual gradient accumulation for larger batch simulation, gradient clipping for stability, and a learning rate scheduler with warm-up and linear decay. It supports validation through optional dataset splitting, robust checkpoint management for progress saving and resumption, and best model selection based on validation loss.

## Features

### Device Selection
- Automatically detects and uses the most suitable hardware:
  - **NVIDIA GPU**: Prioritized if CUDA is available.
  - **Intel OpenVINO**: Attempts to use if installed and compatible.
  - **CPU**: Used as a fallback if no GPU is available.

### Tokenizer and Model Customization
- Automatically adds a special padding token (`[PAD]`) to the tokenizer.
- Resizes the model's token embeddings to match the updated tokenizer.

### Training Enhancements
- **Manual Gradient Accumulation**:
  - Accumulates gradients over multiple steps to simulate larger batch sizes.
- **Gradient Clipping**:
  - Ensures stable training by limiting gradient norms.
- **Learning Rate Scheduler**:
  - Implements a warm-up phase followed by linear decay for smoother training.
- **Validation Support**:
  - Optionally splits the dataset into training and validation sets for performance monitoring.

### Checkpoint Management
- **Checkpoint Saving**:
  - Periodically saves model and optimizer states, along with training progress.
- **Checkpoint Loading**:
  - Supports resuming training from the latest or best checkpoint.
- **Best Model Selection**:
  - Tracks validation loss to identify and save the best-performing model.

### Logging
- Provides detailed logs:
  - Device selection and configuration.
  - Training progress, including loss and learning rates.
  - Checkpoint save/load status.

### Model Saving
- Saves the trained model and tokenizer for later use.
- Ensures compatibility with the Hugging Face Transformers library.

## Requirements
- Python 3.7+
- PyTorch
- Transformers library (Hugging Face)
- Optional: Intel OpenVINO runtime

## Project Structure

```
gpt2-training-project
├── src
│   ├── train.py          # Main script for training the GPT-2 model
│   ├── model.py          # Defines the GPT-2 model architecture
│   └── utils.py          # Utility functions for data preprocessing and evaluation
├── data
│   └── dataset.csv       # Dataset used for training the GPT-2 model
├── docker
│   ├── Dockerfile        # Instructions for building the Docker image
│   └── entrypoint.sh     # Entry point script for the Docker container
├── kubernetes
│   ├── deployment.yaml    # Kubernetes deployment configuration
│   └── service.yaml       # Kubernetes service configuration
├── requirements.txt       # Python dependencies for the project
├── .dockerignore          # Files to ignore when building the Docker image
├── .gitignore             # Files to ignore in Git
└── README.md              # Project documentation
```

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

