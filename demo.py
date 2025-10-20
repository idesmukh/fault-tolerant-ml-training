import os
import shutil
from train import train_with_checkpointing

def main():
    checkpoint_dir = './demo_checkpoints'

    print("=" * 60)
    print("Fault-tolerant machine learning training pipeline for solar energy forecasting")
    print("=" * 60)
    print("\nThis is a simple demo.")
    print("When you press Ctrl+C anytime training will resume on restart.\n")
    print("Start of demo.\n")

    try:
        train_with_checkpointing(
            num_epochs=20,
            checkpoint_dir=checkpoint_dir,
            batch_size=32
        )

        print("\n" + "=" * 60)
        print("Training completed successfully.")
        print("=" * 60)
        print("\nRemoving temporary checkpoints...")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print("End of demo.")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted, saving checkpoint...")
        print("=" * 60)
        print(f"\nCheckpoint saved in: {checkpoint_dir}")
        print("Please run 'python demo.py' to resume training.")
        print("=" * 60)

if __name__ == "__main__":
    main()