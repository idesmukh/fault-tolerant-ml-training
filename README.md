# Fault-tolerant machine learning training pipeline for solar power forecasting

I'm excited to release a fault-tolerant machine learning training pipeline that automatically recovers from cloud interruptions. Press Ctrl+C anytime to simulate an unexpected interruption during training, restart the system, and see it automatically resume from the checkpoint where it stopped. Training machine learning models on spot instances allows for up to 90% cost savings [1]. However, spot instances may be interrupted at any time making fault-tolerance essential.

The ML training pipeline is built to forecast solar power generation using an LSTM with synthetic data for demonstration. The entire implementation is less than 500 lines of clean, testable code. It uses Python and PyTorch for model training, and pytest for test-driven development. System design is based on a philosophy of simple interfaces and deep modules [2]. To ensure data reliability, atomic writes are used for checkpointing.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Run the demo with the following terminal command:
```bash
python demo.py
```

Press Ctrl+C at anytime to simulate interruption, then restart to resume automatically.

## Testing
```bash
pytest
```

## Project Structure

- `model.py` - LSTM model and data generation
- `checkpoint.py` - Atomic checkpointing logic
- `train.py` - Training loop with recovery
- `demo.py` - Interactive demonstration

## References

[1] Cast AI, "Reduce cloud costs with spot instances," Cast AI Blog. [Online]. Available: https://lnkd.in/eMFhTzuA. [Accessed: Oct. 20, 2025].

[2] J. Ousterhout, A Philosophy of Software Design. Palo Alto, CA: Yaknyam Press, 2018.