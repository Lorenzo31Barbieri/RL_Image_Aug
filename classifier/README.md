# Classifier Package

CIFAR-10 image classifier training and inference utilities.

## Structure

- `main.py` - Training script for VGG19 classifier on CIFAR-10
- `predict.py` - Inference and prediction utilities

## Training

Train a VGG19 classifier on CIFAR-10:

```bash
cd classifier
python main.py [--lr 0.1] [--resume]
```

### Arguments
- `--lr` - Learning rate (default: 0.1)
- `--resume` - Resume training from checkpoint

### Output
- Trained model saved to `./checkpoint/ckpt.pth`
- Training progress with loss and accuracy metrics
- Best model automatically saved based on validation accuracy

## Prediction

Run inference on test images:

```bash
cd classifier
python predict.py
```

Features:
- Loads trained model from checkpoint
- Displays sample predictions with ground truth
- Calculates overall test accuracy
- Visualizes prediction results

## Model Architecture

Uses VGG19 architecture adapted for CIFAR-10:
- Input: 32x32 RGB images
- Output: 10 classes (CIFAR-10 categories)
- Optimized with SGD and cosine annealing scheduler

## Requirements

- PyTorch
- torchvision
- matplotlib (for prediction visualization)
- CIFAR-10 dataset (downloaded automatically)
