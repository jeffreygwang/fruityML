#hello fruitCNN

CNN for fruit classification. Used on Kaggle.

![fruits_model](https://user-images.githubusercontent.com/39931478/149639238-c5c04c73-f039-4695-b403-d5ce3d5131cf.png)

## Environment

This network uses keras/TensorFlow for the network setup, sk-learn and pandas for pre-processing data, and matplotlib for graphing image samples.

Download the dataset here: https://www.kaggle.com/moltean/fruits. I ran this in a conda environment; see the dependencies in the yml file.

## Architecture

This model borrows heavily from the CNN architecture first laid out in LeNet and ConvNet with some modern optimizations. There are 131 different types of fruit in this dataset, which are classified with the following architecture:

- Images are processed and encoded in one-hot format
- They are first run through three convolutional/max pooling layers
- The results are then flattened, and processed through three fully connected layers. 
- The outputs are classified with a softmax activation function. 

This model also has some modern optimizations: 

- Adam optimization function (instead of gradient descent)
- Dropout Normalization 
- Batch Normalization 
- The fully connected layers have the following structure for each layer: Batch Norm + ReLU Activation + Dropout

## Performance

On a 2019 MacBook Pro (no GPU's or speedups), with 5 epochs of training, this takes about 30 minutes to run and classifies results with 97% accuracy. 
