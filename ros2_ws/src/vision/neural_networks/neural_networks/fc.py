# MOBILE ROBOTS - FI-UNAM, 2026-2
# TRAINING A NEURAL NETWORK (PyTorch, manual backprop)

import cv2
import sys
import random
import os

import torch
import rclpy
from ament_index_python.packages import get_package_share_directory

NAME = "Rocio Fabiola Romero Bernal"


class FCNeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        # layers: [784, n2, n3, ..., nl, 10]
        self.num_layers = len(layers)

        # Inicializamos biases y pesos con PyTorch
        if biases is None:
            self.biases = [torch.randn(y, 1) for y in layers[1:]]
        else:
            self.biases = biases

        if weights is None:
            self.weights = [
                torch.randn(y, x) for x, y in zip(layers[:-1], layers[1:])
            ]
        else:
            self.weights = weights

    def feedforward(self, x):
        # x: (n_entrada, 1)
        y = []
        a = x
        y.append(a)  # "capa 0" = entrada

        for w, b in zip(self.weights, self.biases):
            # u = W * a + b
            u = torch.matmul(w, a) + b
            # activación sigmoide
            a = torch.sigmoid(u)
            y.append(a)

        return y

    def backpropagate(self, x, t):
        # y[k] es la salida de la capa k (y[0] = entrada, y[-1] = salida final)
        y = self.feedforward(x)

        # Inicializamos gradientes con ceros (misma forma que pesos y biases)
        nabla_b = [torch.zeros_like(b) for b in self.biases]
        nabla_w = [torch.zeros_like(w) for w in self.weights]

        # ---- Capa de salida ----
        # delta_L = (y_L - t) * y_L * (1 - y_L)
        delta = (y[-1] - t) * y[-1] * (1.0 - y[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = torch.matmul(delta, y[-2].T)  # y[-2] salida capa anterior

        # ---- Capas ocultas hacia atrás ----
        # Recorremos desde la penúltima capa oculta hasta la primera capa oculta
        for i in range(2, self.num_layers):
            # weights[-i+1] = matriz de pesos de la capa siguiente
            # y[-i]         = salida de la capa actual
            delta = torch.matmul(self.weights[-i + 1].T, delta) * y[-i] * (1.0 - y[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = torch.matmul(delta, y[-i - 1].T)  # y[-i-1] salida anterior

        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        # batch: lista de (x, t)
        batch_nabla_b = [torch.zeros_like(b) for b in self.biases]
        batch_nabla_w = [torch.zeros_like(w) for w in self.weights]

        M = len(batch)

        for x, t in batch:
            if not rclpy.ok():
                break

            nabla_w, nabla_b = self.backpropagate(x, t)

            batch_nabla_w = [
                bnw + nw / M for bnw, nw in zip(batch_nabla_w, nabla_w)
            ]
            batch_nabla_b = [
                bnb + nb / M for bnb, nb in zip(batch_nabla_b, nabla_b)
            ]

        # Actualizamos pesos y biases (este es el entrenamiento)
        self.weights = [w - eta * nw for w, nw in zip(self.weights, batch_nabla_w)]
        self.biases = [b - eta * nb for b, nb in zip(self.biases, batch_nabla_b)]

        return batch_nabla_w, batch_nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([(nw * nw).sum().item() for nw in nabla_w])
        mag_b = sum([(nb * nb).sum().item() for nb in nabla_b])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, len(training_data), batch_size)
            ]

            for batch in batches:
                if not rclpy.ok():
                    return
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                grad_mag = self.get_gradient_mag(nabla_w, nabla_b)
                sys.stdout.write(
                    "\rGradient magnitude: %f " % (grad_mag)
                )
                sys.stdout.flush()

            print("\nEpoch: " + str(j))


# -----------------------------
#  Dataset
# -----------------------------
def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"

    training_dataset, training_labels = [], []
    testing_dataset, testing_labels = [], []

    for i in range(10):
        # Leemos 1000 imágenes por dígito, 784000 bytes
        with open(folder + "data" + str(i), "rb") as f:
            raw = f.read(784000)
        f_data = [c / 255.0 for c in raw]

        # Creamos imágenes como tensores (784,1)
        images = [
            torch.tensor(f_data[784 * j:784 * (j + 1)], dtype=torch.float32).reshape(784, 1)
            for j in range(1000)
        ]

        # Etiqueta one-hot 10x1 como tensor
        label = torch.tensor(
            [1 if i == j else 0 for j in range(10)],
            dtype=torch.float32
        ).reshape(10, 1)

        # Mitad entrenamiento, mitad prueba
        half = len(images) // 2
        training_dataset += images[0:half]
        training_labels += [label for _ in range(half)]

        testing_dataset += images[half:]
        testing_labels += [label for _ in range(half)]

    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))


# -----------------------------
#  main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    print("TRAINING A NEURAL NETWORK (manual, PyTorch tensors) - " + NAME)

    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")

    epochs = 3
    batch_size = 10
    learning_rate = 10.0

    training_dataset, testing_dataset = load_dataset(dataset_folder)

    # Arquitectura
    nn = FCNeuralNetwork([784, 30, 10])

    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)

    print("\nPress key to test network or ESC to exit...")

    cmd = cv2.waitKey(0)

    while cmd != 27 and rclpy.ok():
        img, label = testing_dataset[torch.randint(0, len(testing_dataset), (1,)).item()]

        # y[-1] es la salida final
        y = nn.feedforward(img)[-1].T  # (1,10)
        y_np = y.detach().numpy()      # para imprimir

        print("\nPerceptron output: " + str(y_np))
        print("Expected output : " + str(label.T.numpy()))
        print("Recognized digit : " + str(y_np.argmax()))

        # Mostrar imagen con OpenCV
        img_np = img.detach().numpy().astype("float32").reshape((28, 28, 1))
        cv2.imshow("Digit", img_np)

        cmd = cv2.waitKey(0)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
