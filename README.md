# MNIST Training with PyTorch

Training a CNN model on MNIST dataset using PyTorch targeting 99.4% accuracy with less than 20k parameters in 20 epoch.

## Model Architecture
- Input: MNIST images (28x28)
- Convolutional layers with BatchNorm
- Dropout for regularization (0.05-0.06)
- Global Average Pooling
- Parameters: < 20k

## Training Details
- Optimizer: SGD with momentum (0.9)
- Scheduler: OneCycleLR
  - max_lr: 0.1
  - epochs: 20
  - pct_start: 0.4
- Data Augmentation:
  - Random Affine (±20°, translate=0.1, scale=0.9-1.1, shear=±5°)
  - Color Jitter (brightness=0.2, contrast=0.2)


## Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 32, 24, 24]           4,640
       BatchNorm2d-5           [-1, 32, 24, 24]              64
           Dropout-6           [-1, 32, 24, 24]               0
            Conv2d-7           [-1, 10, 24, 24]             330
       BatchNorm2d-8           [-1, 10, 24, 24]              20
           Dropout-9           [-1, 10, 24, 24]               0
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,456
      BatchNorm2d-12           [-1, 16, 10, 10]              32
          Dropout-13           [-1, 16, 10, 10]               0
           Conv2d-14             [-1, 16, 8, 8]           2,320
      BatchNorm2d-15             [-1, 16, 8, 8]              32
          Dropout-16             [-1, 16, 8, 8]               0
           Conv2d-17             [-1, 16, 6, 6]           2,320
      BatchNorm2d-18             [-1, 16, 6, 6]              32
          Dropout-19             [-1, 16, 6, 6]               0
           Conv2d-20             [-1, 16, 6, 6]           2,320
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
        AvgPool2d-23             [-1, 16, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             170
================================================================

Total params: 13,960
Trainable params: 13,960
Non-trainable params: 0
----------------------------------------------------------------
- Input Shape: 1x28x28
- Output Shape: 10 (MNIST classes)

### Layer-wise Details:
1. Input Layer: 1 channel, 28x28 image
2. First Block:
   - Conv1: 16 filters, 3x3 kernel -> 26x26
   - Conv2: 32 filters, 3x3 kernel -> 24x24
   - Conv3: 10 filters, 1x1 kernel -> 24x24
   - MaxPool: 2x2 stride 2 -> 12x12
3. Second Block:
   - Conv4: 16 filters, 3x3 kernel -> 10x10
   - Conv5: 16 filters, 3x3 kernel -> 8x8
   - Conv6: 16 filters, 3x3 kernel -> 6x6
   - Conv7: 16 filters, 3x3 kernel, padding=1 -> 6x6
4. Final Block:
   - GAP: 6x6 -> 1x1
   - Conv8: 10 filters, 1x1 kernel -> 1x1

## Receptive Field Analysis
Each layer progressively increases the receptive field:
Receptive Field Analysis:
Layer		RF	Stride	Output Size
--------------------------------------------------
conv1		5x5	    1	26x26
conv2		9x9	    1	24x24
conv3		9x9	    1	24x24
pool1		18x18	2	12x12
conv4		26x26	2	10x10
conv5		34x34	2	8x8
conv6		42x42	2	6x6
conv7		50x50	2	6x6
gap		    50x50	2	1x1
conv8		50x50	2	1x1

Final receptive field: 50x50

## Training Details
- Optimizer: SGD with momentum (0.9)
- Learning Rate: OneCycleLR
  - max_lr: 0.1
  - epochs: 20
  - pct_start: 0.4
  - div_factor: 10
  - final_div_factor: 100
- Regularization:
  - BatchNorm after each conv
  - Dropout: 0.07 for first 3 layers, 0.05 for rest
- Data Augmentation:
  - Random Affine (±20°, translate=0.1, scale=0.9-1.1, shear=±5°)
  - Color Jitter (brightness=0.2, contrast=0.2)

## Latest Training Results
<!-- LATEST_RESULT -->
Maximum Accuracy: TBD
<!-- END_LATEST_RESULT -->

## Running Locally