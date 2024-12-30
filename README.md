# MNIST Training with PyTorch

Training a CNN model on MNIST dataset using PyTorch targeting 99.4% accuracy with less than 8k parameters.

## Model Architecture
- Input: MNIST images (28x28)
- Convolutional layers with BatchNorm
- Dropout for regularization (0.05-0.06)
- Global Average Pooling
- Parameters: < 8k

## Training Details
- Optimizer: SGD with momentum (0.9)
- Scheduler: OneCycleLR
  - max_lr: 0.1
  - epochs: 20
  - pct_start: 0.4
- Data Augmentation:
  - Random Affine (±20°, translate=0.1, scale=0.9-1.1, shear=±5°)
  - Color Jitter (brightness=0.2, contrast=0.2)

## Latest Training Results
<!-- LATEST_RESULT -->
Maximum Accuracy: TBD
<!-- END_LATEST_RESULT -->

## Test Results
You can find the test results [here](https://github.com/${{ github.repository }}/actions)

## Running Locally 