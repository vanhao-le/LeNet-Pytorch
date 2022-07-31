# CNN-Pytorch
How to train a CNN network with Pytorch

1. Project structure
.
|__ output
|   |__ model.pth
|   |__ plot.png
|__ model
|   |__ LNet.py
|__ chip  
|   |__ config.py
|__ train.py
|__ inference.py

We have 4 python files here:
- /model/LeNet.py: This is our model architecture based on the LeNet model.
- /chip/config.py: The script uses for configuration.
- train.py: Trains our model on the KMINIT dataset using PyTorch. The trained model will be saved in the /output/model.pth directory.
- inference.py: loads our model and makes predictions on testing images.

2. The pipeline to train the CNN model

2.1. Define our model architechture


2.2. Load the dataset from disk

2.3. Loop over epochs and batches

2.4. Train the model with a forward() step

2.5. Reset gradient zero_gradient(), perform backward() propagation and update the parameters.

The results after 5 epochs training with batch_size 16.

![Alt text](output/train.png?raw=true "The training result.")

