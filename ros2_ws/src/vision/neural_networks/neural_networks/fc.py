
#
# MOBILE ROBOTS - FI-UNAM, 2026-2
# TRAINING A NEURAL NETWORK
#
# Instructions:
# Complete the code to train a neural network for
# handwritten digits recognition.
#
import cv2
import sys
import random
import numpy
import rclpy
from ament_index_python.packages import get_package_share_directory
import os

NAME = "Popoca Zuñiga Daniel Ixbalanque"

class FCNeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        #
        # The list 'layers' indicates the number of neurons in each layer.
        # Remember that the first layer indicates the dimension of the inputs and thus,
        # there is no bias vector fot the first layer.
        # For this practice, 'layers' should be something like [784, n2, n3, ..., nl, 10]
        # All weights and biases are initialized with random values. In each layer we have a matrix
        # of weights where row j contains all the weights of the j-th neuron in that layer. For this example,
        # the first matrix should be of order n2 x 784 and last matrix should be 10 x nl.
        #
        self.num_layers  = len(layers)
        self.biases =[numpy.random.randn(y,1) for y in layers[1:]] if biases == None else biases
        self.weights=[numpy.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])] if weights==None else weights
        
    def feedforward(self, x):
        y = []
        #
        # TODO:
        # Calculate the output of each layer given the input x
        # Return an array y containg the output of each layer
        # You can do the following steps:
        # 
        # append x to y
        # FOR i = [0,..,L):
        #   u = dot product (W[i], x) + B[i]
        #   x = 1.0 / (1.0 + exp(-u)) The output of the i-th layer is the input of the next one
        #   append x to y
        #
        y.append(x)  # Append the input as the output of the first layer
        a = x
        for i in range(self.num_layers - 1):
            z = numpy.dot(self.weights[i], a) + self.biases[i]
            a = 1.0 / (1.0 + numpy.exp(-z))  # Sigmoid activation function
            y.append(a)
        
        return y

    def backpropagate(self, x, t):
        y = self.feedforward(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # TODO:
        # Return a tuple [nabla_w, nabla_b] containing the gradient of cost function J with respect to
        # each weight and bias of all the network. The gradient is calculated assuming only one training
        # example: the input 'x' and the corresponding target 't'.
        # nabla_w and nabla_b should have the same dimensions as the corresponding
        # self.weights and self.biases
        # You can calculate the gradient following these steps:
        #
        # Calculate delta for the output layer L: delta=(y[-1]-t)*y[-1]*(1-y[-1])
        # nabla_b of output layer = delta      
        # nabla_w of output layer = delta*y[-2].T where y[-2].T is the transpose of the ouput vector of layer L-1
        # FOR all layers i=[2,L): 
        #     delta = (W[-i+1].T * delta)*y[-i]*(1 - y[-i])
        #     where 'W[-i+1].T' is the transpose of the matrix of weights of layer -i+1 and 'y[-i]' is the output of layer -i
        #     nabla_b[-i] = delta
        #     nabla_w[-i] = delta*y[-i-1].T  
        #        
        
        # Calculate delta for the output layer
        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)
        
        # Backpropagate through the hidden layers
        for i in range(2, self.num_layers):
            # Current layer index (counting from the end)
            layer_idx = -i
            # Previous delta (from the layer closer to the output)
            prev_delta = delta
            # Weights of the next layer (closer to output)
            next_weights = self.weights[layer_idx + 1]
            # Calculate new delta
            delta = numpy.dot(next_weights.T, prev_delta) * y[layer_idx] * (1 - y[layer_idx])
            # Update gradients
            nabla_b[layer_idx] = delta
            nabla_w[layer_idx] = numpy.dot(delta, y[layer_idx - 1].T)

        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        #
        # This function exectutes gradient descend for the subset of examples
        # given by 'batch' with learning rate 'eta'
        # 'batch' is a list of training examples [(x,t), ..., (x,t)]
        #
        batch_nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        batch_nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x,t in batch:
            if not rclpy.ok():
                break
            nabla_w, nabla_b = self.backpropagate(x,t)
            batch_nabla_w = [bnw+nw/M for bnw,nw in zip(batch_nabla_w, nabla_w)]
            batch_nabla_b = [bnb+nb/M for bnb,nb in zip(batch_nabla_b, nabla_b)]
        self.weights = [w-eta*nw for w,nw in zip(self.weights, batch_nabla_w)]  #This lines are the actual training
        self.biases  = [b-eta*nb for b,nb in zip(self.biases , batch_nabla_b)]
        return batch_nabla_w, batch_nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([numpy.sum(n) for n in [nw*nw for nw in nabla_w]])
        mag_b = sum([numpy.sum(b) for b in [nb*nb for nb in nabla_b]])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,len(training_data), batch_size)]
            for batch in batches:
                if not rclpy.ok():
                    return
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
                sys.stdout.flush()
            print("Epoch: " + str(j))
    #
    ### END OF CLASS
    #


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [],[],[],[]
    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])
        training_dataset += images[0:len(images)//2]
        training_labels  += [label for j in range(len(images)//2)]
        testing_dataset  += images[len(images)//2:len(images)]
        testing_labels   += [label for j in range(len(images)//2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def main(args=None):
    rclpy.init(args=args)
    print("TRAINING A NEURAL NETWORK - " + NAME)
    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")
    
    epochs        = 3
    batch_size    = 10
    learning_rate = 0.5
    training_dataset, testing_dataset = load_dataset(dataset_folder)
    nn = FCNeuralNetwork([784,30,10])
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)

    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind':"{:.3f}".format})
    cmd = cv2.waitKey(0)
    while cmd != 27 and rclpy.ok():
        img,label = testing_dataset[numpy.random.randint(0, 4999)]
        y = nn.feedforward(img)[-1].T
        print("\nPerceptron output: " + str(y))
        print("Expected output  : "   + str(label.T))
        print("Recognized digit : "   + str(numpy.argmax(y)))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28,28,1)))
        cmd = cv2.waitKey(0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()


   
"""
### Programa para hiperparametros ###


#
# MOBILE ROBOTS - FI-UNAM, 2026-2
# TRAINING A NEURAL NETWORK (AUTOMATED EXPERIMENT GRID)
#

import sys
import random
import numpy
import rclpy
import time
import csv
from ament_index_python.packages import get_package_share_directory
import os

NAME = "Popoca Zuñiga Daniel Ixbalanque"


class FCNeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers = len(layers)
        self.biases = [numpy.random.randn(y, 1) for y in layers[1:]] if biases is None else biases
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])] if weights is None else weights

    def feedforward(self, x):
        y = []
        y.append(x)
        a = x
        for i in range(self.num_layers - 1):
            z = numpy.dot(self.weights[i], a) + self.biases[i]
            a = 1.0 / (1.0 + numpy.exp(-z))
            y.append(a)
        return y

    def backpropagate(self, x, t):
        y = self.feedforward(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)

        for i in range(2, self.num_layers):
            prev_delta = delta
            next_weights = self.weights[-i + 1]
            delta = numpy.dot(next_weights.T, prev_delta) * y[-i] * (1 - y[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = numpy.dot(delta, y[-i - 1].T)

        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        batch_nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        batch_nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x, t in batch:
            if not rclpy.ok():
                break
            nabla_w, nabla_b = self.backpropagate(x, t)
            batch_nabla_w = [bnw + nw / M for bnw, nw in zip(batch_nabla_w, nabla_w)]
            batch_nabla_b = [bnb + nb / M for bnb, nb in zip(batch_nabla_b, nabla_b)]

        self.weights = [w - eta * nw for w, nw in zip(self.weights, batch_nabla_w)]
        self.biases  = [b - eta * nb for b, nb in zip(self.biases, batch_nabla_b)]

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                if not rclpy.ok():
                    return
                self.update_with_batch(batch, eta)

def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])
        training_dataset += images[:500]
        training_labels  += [label] * 500
        testing_dataset  += images[500:]
        testing_labels   += [label] * 500
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def evaluate_accuracy(nn, testing_dataset, num_tests=100):
    correct = 0
    for _ in range(num_tests):
        img, label = testing_dataset[numpy.random.randint(0, len(testing_dataset))]
        y = nn.feedforward(img)[-1]
        if numpy.argmax(y) == numpy.argmax(label):
            correct += 1
    return correct / num_tests


def run_experiment(lr, ep, bs, training_dataset, testing_dataset):
    nn = FCNeuralNetwork([784, 30, 10])

    t0 = time.time()
    nn.train_by_SGD(training_dataset, ep, bs, lr)
    training_time = time.time() - t0

    success_rate = evaluate_accuracy(nn, testing_dataset, num_tests=100)

    return training_time, success_rate


def run_all_experiments(training_dataset, testing_dataset):
    learning_rates = [0.5, 1.0, 3.0, 10.0]
    epochs_list     = [3, 10, 50, 100]
    batch_sizes     = [5, 10, 30, 100]

    results = []

    for lr in learning_rates:
        for ep in epochs_list:
            for bs in batch_sizes:

                print(f"\nRunning experiment LR={lr}, Epochs={ep}, Batch={bs}...")

                training_time, success_rate = run_experiment(
                    lr, ep, bs, training_dataset, testing_dataset
                )

                results.append({
                    "learning_rate": lr,
                    "epochs": ep,
                    "batch_size": bs,
                    "training_time": training_time,
                    "por_successes": success_rate
                })

                print(f"Finished → time={training_time:.2f}s,  accuracy={success_rate:.3f}")

    return results


def save_results_to_csv(results, filename="experiments.csv"):
    keys = results[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)



def main(args=None):
    rclpy.init(args=args)
    print("RUNNING 64 EXPERIMENT GRID - " + NAME)

    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")

    training_dataset, testing_dataset = load_dataset(dataset_folder)

    print("\nStarting all 64 experiments...")
    results = run_all_experiments(training_dataset, testing_dataset)

    save_results_to_csv(results)
    print("\nAll experiments completed!")
    print("Results saved in experiments.csv")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
"""


"""
# Matriz de confusion 

#
# MOBILE ROBOTS - FI-UNAM, 2026-2
# TRAINING A NEURAL NETWORK
#

import cv2
import sys
import random
import numpy
import rclpy
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ament_index_python.packages import get_package_share_directory
import os

NAME = "Popoca Zuñiga Daniel Ixbalanque"

class FCNeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers  = len(layers)
        self.biases =[numpy.random.randn(y,1) for y in layers[1:]] if biases is None else biases
        self.weights=[numpy.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])] if weights is None else weights
        
    def feedforward(self, x):
        y = []
        y.append(x)
        a = x
        for i in range(self.num_layers - 1):
            z = numpy.dot(self.weights[i], a) + self.biases[i]
            a = 1.0 / (1.0 + numpy.exp(-z))
            y.append(a)
        return y

    def backpropagate(self, x, t):
        y = self.feedforward(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)
        
        for i in range(2, self.num_layers):
            layer_idx = -i
            prev_delta = delta
            next_weights = self.weights[layer_idx + 1]
            delta = numpy.dot(next_weights.T, prev_delta) * y[layer_idx] * (1 - y[layer_idx])
            nabla_b[layer_idx] = delta
            nabla_w[layer_idx] = numpy.dot(delta, y[layer_idx - 1].T)

        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        batch_nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        batch_nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)

        for x,t in batch:
            if not rclpy.ok():
                break
            nabla_w, nabla_b = self.backpropagate(x,t)
            batch_nabla_w = [bnw + nw/M for bnw, nw in zip(batch_nabla_w, nabla_w)]
            batch_nabla_b = [bnb + nb/M for bnb, nb in zip(batch_nabla_b, nabla_b)]

        self.weights = [w - eta*nw for w,nw in zip(self.weights, batch_nabla_w)]
        self.biases  = [b - eta*nb for b,nb in zip(self.biases , batch_nabla_b)]

        return batch_nabla_w, batch_nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([numpy.sum(n) for n in [nw * nw for nw in nabla_w]])
        mag_b = sum([numpy.sum(b) for b in [nb * nb for nb in nabla_b]])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        error_hist = []
        time_hist = []
        t0 = time.time()

        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,len(training_data), batch_size)]

            for batch in batches:
                if not rclpy.ok():
                    break
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                grad_mag = self.get_gradient_mag(nabla_w, nabla_b)

                error_hist.append(grad_mag)
                time_hist.append(time.time() - t0)

                sys.stdout.write("\rGradient magnitude: %f" % grad_mag)
                sys.stdout.flush()

            print("\nEpoch:", j)

        return error_hist, time_hist


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [],[],[],[]

    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])

        training_dataset += images[0:500]
        training_labels  += [label for j in range(500)]
        testing_dataset  += images[500:1000]
        testing_labels   += [label for j in range(500)]

    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusión - Red Neuronal")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    print("TRAINING A NEURAL NETWORK - " + NAME)

    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")

    epochs        = 3
    batch_size    = 10
    learning_rate = 10.0

    training_dataset, testing_dataset = load_dataset(dataset_folder)
    nn = FCNeuralNetwork([784, 30, 10])

    # ENTRENAMIENTO
    error_hist, time_hist = nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)

    # PREDICCIONES PARA MATRIZ DE CONFUSIÓN
    y_true = []
    y_pred = []

    for img, label in testing_dataset:
        out = nn.feedforward(img)[-1]
        y_pred.append(int(numpy.argmax(out)))
        y_true.append(int(numpy.argmax(label)))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # ==============================
    # 1) ERROR VS TIEMPO
    # ==============================
    plt.figure(figsize=(10,5))
    plt.plot(time_hist, error_hist, linewidth=2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Magnitud del Gradiente (Error)")
    plt.title("Error vs Tiempo durante el Entrenamiento")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==============================
    # 2) MATRIZ DE CONFUSIÓN
    # ==============================
    plot_confusion_matrix(cm)

    # TEST INTERACTIVO
    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind':"{:.3f}".format})
    cmd = cv2.waitKey(0)

    while cmd != 27 and rclpy.ok():
        img, label = testing_dataset[numpy.random.randint(0, 499)]
        y = nn.feedforward(img)[-1].T
        print("\nPerceptron output:", y)
        print("Expected output  :", label.T)
        print("Recognized digit :", numpy.argmax(y))

        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28,28,1)))
        cmd = cv2.waitKey(0)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
"""