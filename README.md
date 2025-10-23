# Fault-tolerant machine learning training pipeline for solar power forecasting

The availability of compute for training machine learning (ML) models has grown exponentially. However, managing ML training costs remains important. One way to ensure low training costs is by designing effective ML training pipelines. Usage of spot instances can allow for up to 90% cost savings compared to on-demand instances [1]. However, spot instances may be interrupted at any time making fault-tolerance essential. I have therefore implemented a production-ready ML training pipeline for solar power forecasting that can be used for training on spot instances.

## Features

The ML training pipeline is built to forecast solar power generation using an LSTM with synthetic data for demonstration. The entire implementation is less than 500 lines of clean, testable code. It uses Python and PyTorch for model training, and pytest for test-driven development. System design is based on a philosophy of simple interfaces and deep modules [2]. To ensure data reliability, atomic writes are used for checkpointing.

In case of an interruption during the training, a checkpoint of the model including weights, optimizer state, batch index, epoch, step and loss, are saved in a file. When the system is restarted, it automatically resumes from the previous checkpoint. This ML training pipeline therefore allows one to use spot instances for training without losing training progress.

## Getting started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/idesmukh/fault-tolerant-ml-training
    cd systemeye
    ```

2.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the demo from your terminal with the following command:
```bash
python3 demo.py
```

Press Ctrl+C at anytime to simulate interruption, then restart to resume automatically.

### Testing
```bash
python3 -m pytest
```

## Project structure

- `model.py` - LSTM model and synthetic data generation
- `checkpoint.py` - Checkpointing
- `train.py` - Training loop
- `demo.py` - Demonstration

## References

[1] Cast AI, "Reduce cloud costs with spot instances," Cast AI Blog. [Online]. Available: https://cast.ai/blog/reduce-cloud-costs-with-spot-instances/. [Accessed: Oct. 20, 2025].

[2] J. Ousterhout, A Philosophy of Software Design. Palo Alto, CA: Yaknyam Press, 2018.

## License

This code is provided under the MIT license.