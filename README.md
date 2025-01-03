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


<!-- LATEST_RESULT -->
Maximum Accuracy: TBD
99.53%

## Training/Testing Logs


Model Parameters: 13960
Model Architecture:
Net(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (bn3): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv5): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv7): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn7): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (gap): AvgPool2d(kernel_size=6, stride=6, padding=0)
  (conv8): Conv2d(16, 10, kernel_size=(1, 1), stride=(1, 1))
  (dropout): Dropout(p=0.05, inplace=False)
  (dropout1): Dropout(p=0.07, inplace=False)
  (dropout2): Dropout(p=0.07, inplace=False)
  (dropout3): Dropout(p=0.01, inplace=False)
  (dropout4): Dropout(p=0.05, inplace=False)
  (dropout5): Dropout(p=0.05, inplace=False)
  (dropout6): Dropout(p=0.05, inplace=False)
  (dropout7): Dropout(p=0.01, inplace=False)
  (dropout8): Dropout(p=0.005, inplace=False)
)

Receptive Field Analysis:
Layer		RF	Stride	Output Size
--------------------------------------------------
conv1		5x5	1	26x26
conv2		9x9	1	24x24
conv3		9x9	1	24x24
pool1		18x18	2	12x12
conv4		26x26	2	10x10
conv5		34x34	2	8x8
conv6		42x42	2	6x6
conv7		50x50	2	6x6
gap		50x50	2	1x1
conv8		50x50	2	1x1

Final receptive field: 50x50
Epoch : 1 loss=0.11583685874938965 accuracy=79.6 batch_id=468: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:19<00:00, 24.39it/s]

Test set: Average loss: 0.0614, Accuracy: 9832.0/10000 (98.32%)

Epoch : 2 loss=0.08818384259939194 accuracy=96.38833333333334 batch_id=468: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.12it/s]

Test set: Average loss: 0.0535, Accuracy: 9825.0/10000 (98.25%)

Epoch : 3 loss=0.03574211522936821 accuracy=97.195 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.97it/s]

Test set: Average loss: 0.0303, Accuracy: 9904.0/10000 (99.04%)

Epoch : 4 loss=0.08418970555067062 accuracy=97.43333333333334 batch_id=468: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.12it/s]

Test set: Average loss: 0.0255, Accuracy: 9911.0/10000 (99.11%)

Epoch : 5 loss=0.07115250080823898 accuracy=97.74666666666667 batch_id=468: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.13it/s]

Test set: Average loss: 0.0259, Accuracy: 9910.0/10000 (99.10%)

Epoch : 6 loss=0.0715542733669281 accuracy=97.87833333333333 batch_id=468: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.03it/s]

Test set: Average loss: 0.0242, Accuracy: 9925.0/10000 (99.25%)

Epoch : 7 loss=0.0496901273727417 accuracy=97.97333333333333 batch_id=468: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.17it/s]

Test set: Average loss: 0.0244, Accuracy: 9911.0/10000 (99.11%)

Epoch : 8 loss=0.06228545680642128 accuracy=98.12833333333333 batch_id=468: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.88it/s]

Test set: Average loss: 0.0207, Accuracy: 9925.0/10000 (99.25%)

Epoch : 9 loss=0.03513186797499657 accuracy=98.17166666666667 batch_id=468: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.78it/s]

Test set: Average loss: 0.0231, Accuracy: 9921.0/10000 (99.21%)

Epoch : 10 loss=0.07453744858503342 accuracy=98.16166666666666 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.14it/s]

Test set: Average loss: 0.0222, Accuracy: 9925.0/10000 (99.25%)

Epoch : 11 loss=0.10506059974431992 accuracy=98.29666666666667 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.21it/s]

Test set: Average loss: 0.0191, Accuracy: 9942.0/10000 (99.42%)

Accuracy is greater than 99.4 and reached at epoch 11
Epoch : 12 loss=0.02676127292215824 accuracy=98.31833333333333 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.90it/s]

Test set: Average loss: 0.0191, Accuracy: 9936.0/10000 (99.36%)

Epoch : 13 loss=0.12022185325622559 accuracy=98.425 batch_id=468: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.84it/s]

Test set: Average loss: 0.0203, Accuracy: 9928.0/10000 (99.28%)

Epoch : 14 loss=0.08535778522491455 accuracy=98.39833333333333 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.96it/s]

Test set: Average loss: 0.0172, Accuracy: 9953.0/10000 (99.53%)

Accuracy is greater than 99.4 and reached at epoch 14
Epoch : 15 loss=0.018056271597743034 accuracy=98.51833333333333 batch_id=468: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.06it/s]

Test set: Average loss: 0.0161, Accuracy: 9946.0/10000 (99.46%)

Accuracy is greater than 99.4 and reached at epoch 15
Epoch : 16 loss=0.047863736748695374 accuracy=98.485 batch_id=468: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.73it/s]

Test set: Average loss: 0.0191, Accuracy: 9938.0/10000 (99.38%)

Epoch : 17 loss=0.006022705230861902 accuracy=98.47166666666666 batch_id=468: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.23it/s]

Test set: Average loss: 0.0176, Accuracy: 9942.0/10000 (99.42%)

Accuracy is greater than 99.4 and reached at epoch 17
Epoch : 18 loss=0.11224853992462158 accuracy=98.47666666666667 batch_id=468: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 27.18it/s]

Test set: Average loss: 0.0170, Accuracy: 9943.0/10000 (99.43%)

Accuracy is greater than 99.4 and reached at epoch 18
Epoch : 19 loss=0.007370114326477051 accuracy=98.55666666666667 batch_id=468: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:17<00:00, 26.12it/s]

Test set: Average loss: 0.0174, Accuracy: 9938.0/10000 (99.38%)

Maximum Accuracy achieved: 99.53% at epoch 14



