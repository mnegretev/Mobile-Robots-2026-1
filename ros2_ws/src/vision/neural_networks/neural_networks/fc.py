#
# MOBILE ROBOTS - FI-UNAM, 2026-2
# TRAINING A NEURAL NETWORK (PyTorch version)
#
#

import cv2
import sys
import random
import numpy
import rclpy
from ament_index_python.packages import get_package_share_directory
import os

import torch
import torch.nn as nn
import torch.optim as optim

NAME = "Rocio"


class FCNeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.device = torch.device("cpu") 

        modules = []
        for in_f, out_f in zip(layers[:-1], layers[1:]):
            # Capa lineal
            modules.append(nn.Linear(in_f, out_f))
            # Capa de activación sigmoide
            modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules).to(self.device)
        self.loss_fn = nn.MSELoss()

    def train_by_SGD(self, training_data, epochs, batch_size, eta):

        # Optimizador con tasa de aprendizaje eta
        optimizer = optim.SGD(self.model.parameters(), lr=eta)

        for epoch in range(epochs):
            random.shuffle(training_data)

            # partimos en batches
            batches = [training_data[k:k + batch_size]
                       for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                if not rclpy.ok():
                    return

                # Construir batch como tensores de PyTorch
                # x: (784,1) -> hstack -> (784, M) -> transpose -> (M, 784)
                xs = numpy.hstack([x for (x, _) in batch]).T
                ts = numpy.hstack([t for (_, t) in batch]).T

                xs = xs.astype("float32")
                ts = ts.astype("float32")

                inputs = torch.from_numpy(xs).to(self.device)    # shape (M, 784)
                targets = torch.from_numpy(ts).to(self.device)   # shape (M, 10)

                # Forward + backward + update
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(inputs)          # (M, 10)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                # Mostrar pérdida en consola
                sys.stdout.write("\rLoss: %f            " % loss.item())
                sys.stdout.flush()

            print("\nEpoch: " + str(epoch))

    def predict(self, x_np):
        self.model.eval()
        # (784,1) -> (1,784)
        x_flat = x_np.reshape(1, -1).astype("float32")
        x_tensor = torch.from_numpy(x_flat).to(self.device)

        with torch.no_grad():
            y_tensor = self.model(x_tensor)  

        y_np = y_tensor.cpu().numpy().T  
        return y_np


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c / 255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784 * j:784 * (j + 1)], dtype="float32").reshape([784, 1])
                  for j in range(1000)]
        label = numpy.asarray([1 if i == j else 0 for j in range(10)],
                              dtype="float32").reshape([10, 1])
        training_dataset += images[0:len(images) // 2]
        training_labels += [label for _ in range(len(images) // 2)]
        testing_dataset += images[len(images) // 2:len(images)]
        testing_labels += [label for _ in range(len(images) // 2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))


def main(args=None):
    rclpy.init(args=args)
    print("TRAINING A NEURAL NETWORK (PyTorch) - " + NAME)
    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")

 
    epochs = 5           # número de épocas
    batch_size = 10      # tamaño de batch
    learning_rate = 10   
    # <<< ---------------------------------------- >>>

    training_dataset, testing_dataset = load_dataset(dataset_folder)

    # Red: 784 -> 30 -> 10 con lineal/sigmoide alternadas
    nn_torch = FCNeuralNetwork([784, 30, 10])

    # Entrenar
    nn_torch.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)

    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind': "{:.3f}".format})
    cmd = cv2.waitKey(0)

    while cmd != 27 and rclpy.ok():
        img, label = testing_dataset[numpy.random.randint(0, 4999)]
        y = nn_torch.predict(img).T  
        print("\nPerceptron output: " + str(y))
        print("Expected output  : " + str(label.T))
        print("Recognized digit : " + str(numpy.argmax(y)))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28, 28, 1)))
        cmd = cv2.waitKey(0)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
