This repository contains the Python implementation of regression models for estimating the unified functional spectra based on convolutional neural network (CNN). The CNN regressor estimates the unified functional spectra from the behavioral data of the participants. Separate CNN regressors are constructed to estimate the spectrum intensities for different functional axis.

We constructed two types of CNN models, called Early-fusion and Late-fusion. Both models constructed for each basis consists of three convolutional layers (Conv1, Conv2, and Conv3), two max-pooling layers (MaxPooling1 and MaxPooling2), and two fully connected layers (FC4 and FC5). ReLU activation was applied to the convolutional layers and FC4, while the output layer FC5 used sigmoid activation. To stabilize the training process and mitigate overfitting, Batch Normalization and Dropout were applied after each layer, and L2 regularization was introduced to the convolutional and fully connected layers.

This repository contains the following seven files in addition to the README.

- **1-1.preprocess_feature.py**  
It extracts the behavioral data (head pose angular velocity, utterance status, gaze status, eyeball direction, and facial expressions) and performs preprocessing. 
Except for the utterance status, the input features are obtained from CSV files extracted using OpenFace.
The head pose angular velocities were calculated by the frame difference of the head pose angles between adjacent frames. They are clipped to [-200, 200] and normalized to the range between -1 and 1. 
The utterance status is the temporally smoothed sequence using a moving average window (50 frames long) applied to a binary sequence indicating the framewise presence or absence of an utterance obtained from a manual transcript. 
The eyeball direction is defined as the horizontal angle of the eyeball relative to the frontal facial direction. This value is computed from the OpenFace CSV outputs, clipped to the range [−20, 20], and normalized to [−1, 1].
Facial expressions were the intensities of 17 action units and were normalized to the range between 0 and 1.
- **1-2.preprocess_label.py**  
It extracts the unified functional spectra and normalized to the range between 0 and 1.
- **2.shape_data.py**  
It shape the behavioral data into inputs to the regression model as short time series segments of approximately 1 second (32 frames), centered on the target frame to be predicted.
- **3.make_dataset.py**  
It integrates and splits the session-wise feature and label data created in 2.shape_data.py by group, and creates test, training, and validation datasets using a Leave-One-Group-Out (LOGO) scheme.
- **4.training_and_prediction.py**  
It trains a model using the dataset created in 3.make_dataset.py and then performs predictions with the trained model.
The training dataset is balanced by random oversampling or undersampling so that the numbers of above-average and below-average samples were equal.
The model is trained on randomly sampled training data and predicts on a test set. 
During prediction, the model outputs the unified functional spectrum as continuous values in the range of 0 to 1.
Training is terminated if the validation loss failed to improve by at least 0.001 within the most recent 10 epochs.
- **model.py**  
It is used in 4.training_and_prediction.py. It contains the model configuration.
- **path.py**  
It specifies the file paths for the input features and the unified functional spectrum label data used in the preprocessing scripts 1-1 and 1-2.

When using for the first time, execute the Python files 1-1, 1-2, 2, 3, and 4 in sequence.

The following summarizes the computational resources, libraries, and parameters.

- **Computational resources**  
CPU: Intel Core i7-11700  
GPU: NVIDIA GeForce RTX 3080  
Memory: 32GB  

- **Libraries**  
python 3.8.11  
numpy 1.19.5  
scikit-learn 0.24.2  
tensorflow 2.6.0  
keras 2.6.0  

- **Paremeters**  
Random seed: 42  
Input shape: (32,96,1)  
  - Conv1  
Kernel size: Early-fusion (5,96) / Late-fusion (5,1)  
Number of filters: 20  
Kernel regularizer: L2 normalization (weight 0.001)  
  - MaxPooling1  
Kernel size: (4,1)  
Strides: (2,1)  
  - Batch normalization layer  
Momentum: 0.99  
Epsilon: 0.001  
  - Dropout layer  
Rate: 0.1  
  - Conv2  
Kernel size: (6,1)  
Number of filters: 20  
Kernel regularizer: L2 normalization (weight 0.001)  
  - MaxPooling2  
Kernel size: (4,1)  
Strides: (2,1)  
  - Batch normalization layer  
Momentum: 0.99  
Epsilon: 0.001  
  - Dropout layer  
Rate: 0.2  
  - Conv3  
Kernel size: (3,1)  
Number of filters: 10  
Kernel regularizer: L2 normalization (weight 0.001)  
  - Batch normalization layer  
Momentum: 0.99  
Epsilon: 0.001  
  - Dropout layer  
Rate: 0.25  
  - FC4  
Units: 10  
Kernel regularizer: L2 normalization (weight 0.001)  
  - Batch normalization layer  
Momentum: 0.99  
Epsilon: 0.001  
  - FC5  
Units (Output shape): 1  
Kernel regularizer: L2 normalization (weight 0.001)  

  Batchsize: 12  
  Loss function: MAE  
  Optimizer: Adadelta  
  Learning rate: 1.0  
  Decay rate: 0.95  
  Maximum epochs: 100  
