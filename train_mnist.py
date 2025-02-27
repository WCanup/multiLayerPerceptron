import kagglehub
import random
import matplotlib.pyplot as plt
import numpy as np
import struct
from array import array
from os.path import join
from sklearn.model_selection import train_test_split
from mlp import *

# Download latest version
# path = kagglehub.dataset_download("hojjatk/mnist-dataset")

# print("Path to dataset files:", path)

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
#
# Verify Reading Dataset via MnistDataloader class
#

#
# Set file paths based on added MNIST Datasets
#
input_path = 'C:/Users/willc/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(train_x, train_y), (test_x, test_y) = mnist_dataloader.load_data()


# Flatten the 28x28 images into vectors of length 784
train_x = np.array(train_x).reshape(-1, 28*28)
test_x = np.array(test_x).reshape(-1, 28*28)

# Normalize the pixel values to the range [0, 1]
train_x = train_x / 255.0
test_x = test_x / 255.0

# Split the training set into 80% training and 20% validation
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Convert labels to numpy arrays
train_y = np.array(train_y)
val_y = np.array(val_y)
test_y = np.array(test_y)

lossf = CrossEntropy()

#Dataset has 10 numbers 0,1,2,3,4,5,6,7,8,9
num_classes = 10

train_y = lossf.one_hot_encode(train_y, num_classes)
val_y = lossf.one_hot_encode(val_y, num_classes)
test_y = lossf.one_hot_encode(test_y, num_classes)


# Print shapes to verify
print(f"x_train shape: {train_x.shape}")
print(f"x_val shape: {val_x.shape}")
print(f"x_test shape: {test_x.shape}")


layer1 = Layer(fan_in=784, fan_out=128, activation_function=Linear())
layer2 = Layer(fan_in=128, fan_out=64, activation_function=Relu())
layer3 = Layer(fan_in=64, fan_out=10, activation_function=Softmax())

multiLayerPerceptron = MultilayerPerceptron(layers=(layer1, layer2, layer3))
trainingLoss, validationLoss = multiLayerPerceptron.train(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, loss_func=CrossEntropy(), learning_rate=1E-4, batch_size=16, epochs=100)


#Get mean values of losses to plot
predictionT = multiLayerPerceptron.forward(test_x)
testLosses = lossf.loss(y_pred=predictionT, y_true=test_y)
testLoss = np.mean(testLosses)
print("Final Test Loss: {}".format(testLoss))


#Calculate accuracy
predicted_labels = np.argmax(predictionT, axis=1)
true_labels = np.argmax(test_y, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Select 1 sample for each class (0-9) from testing and show these samples along with the predicted class for each
samples = []
for i in range(10):
    idx = np.where(true_labels == i)[0][0]
    samples.append((test_x[idx].reshape(28, 28), predicted_labels[idx]))

# Show the selected samples
plt.figure(figsize=(10, 10))
show_images([s[0] for s in samples], [f"Predicted: {s[1]}" for s in samples])
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(trainingLoss, label='Training', color='b')
plt.plot(validationLoss, label='Validation', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve', size=16)
plt.legend()
plt.show()