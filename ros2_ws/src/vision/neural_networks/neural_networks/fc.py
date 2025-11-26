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

import time   # Para medir el tiempo que tarda en entrenar la red

NAME = "OSCAR CORTES CALDERON"

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
        
        # salida de la capa de entrada
        y.append(x)

        # recorrer todas las capas ocultas y de salida
        for w, b in zip(self.weights, self.biases):
            u = numpy.dot(w, x) + b          # W·x + b
            x = 1.0 / (1.0 + numpy.exp(-u))  # sigmoide
            y.append(x)
        
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
        
        # --- capa de salida ---
        delta = (y[-1] - t) * y[-1] * (1.0 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)

        # --- capas ocultas hacia atrás ---
        for i in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-i+1].T, delta) * y[-i] * (1.0 - y[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = numpy.dot(delta, y[-i-1].T)
        
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

def run_experiment(training_dataset, testing_dataset, epochs, batch_size, learning_rate, n_tests=100, architecture=[784, 30, 10]):
    """
    Se ejecuta un solo experimento con una configuración de:
    - epochs
    - batch_size
    - learning_rate
    y una arquitectura dada.

    Regresa:
    - t_train: tiempo de entrenamiento en segundos
    - accuracy: porcentaje de aciertos (entre 0 y 1) en n_tests clasificaciones
    """
    # Creamos una nueva red con la arquitectura indicada.
    # Aquí es donde cambiá la arquitectura para cumplir la parte de "repetir los experimentos con una arquitectura diferente".
    nn = FCNeuralNetwork(architecture)

    # Entrenamos midiendo el tiempo total de entrenamiento
    t0 = time.time()
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    t_train = time.time() - t0

    # Evaluamos la red haciendo n_tests clasificaciones aleatorias
    aciertos = 0
    n_tests = min(n_tests, len(testing_dataset))  # por seguridad
    for _ in range(n_tests):
        img, label = testing_dataset[numpy.random.randint(0, len(testing_dataset))]
        y = nn.feedforward(img)[-1]  # salida de la última capa
        if numpy.argmax(y) == numpy.argmax(label):
            aciertos += 1

    accuracy = aciertos / n_tests
    return t_train, accuracy

def main(args=None):
    rclpy.init(args=args)
    print("TRAINING A NEURAL NETWORK - " + NAME)
    package_path = get_package_share_directory("neural_networks")
    dataset_folder = os.path.join(package_path, "dataset")
    
    
    # ============================================================================================
    # Selector de modo de trabajo
    # ===========================
    # False -> modo DEMO, es decir, ignora este if y se corre como el código original 
    # True  -> modo EXPERIMENTOS (recorre parámetros para el punto 5 de la practica 4)
    # En este caso lo dejare en True para la realizacion de la practica 
    modo_experimentos = True

    # Cargamos el dataset una sola vez y lo reutilizamos
    training_dataset, testing_dataset = load_dataset(dataset_folder)
    
    if modo_experimentos:
        # ===========================
        # MODO EXPERIMENTOS (PUNTO 5)
        # ===========================
        # Listas de parámetros que pide la práctica:
        tasas_aprendizaje = [0.5, 1.0, 3.0, 10.0]
        epocas_list       = [3, 10, 50, 100]
        batch_sizes       = [5, 10, 30, 100]

        # Aquí guardaremos los resultados para luego hacer tablas / gráficas
        resultados = []

        # Recorremos TODAS las combinaciones de parámetros
        for lr in tasas_aprendizaje:
            for ep in epocas_list:
                for bs in batch_sizes:
                    print(f"\n=== Experimento: lr={lr}, epochs={ep}, batch_size={bs} ===")

                    # Ejecutamos un experimento y obtenemos tiempo y accuracy
                    t_train, acc = run_experiment(
                        training_dataset,
                        testing_dataset,
                        epochs=ep,
                        batch_size=bs,
                        learning_rate=lr,
                        n_tests=100,              # mínimo 100 pruebas de clasificación
                        architecture=[784, 30, 10]  # cambia esta lista para otra arquitectura
                    )

                    print(f"Tiempo de entrenamiento: {t_train:.2f} s")
                    print(f"Porcentaje de éxitos : {acc*100:.2f} %")

                    resultados.append((lr, ep, bs, t_train, acc))

        # Aquí podrías después escribir 'resultados' a un CSV si quieres.
        # Por ahora solo terminamos el nodo.
        rclpy.shutdown()
        return
    # ============================================================================================
    
    
    epochs        = 3
    batch_size    = 10
    learning_rate = 0.5
    #training_dataset, testing_dataset = load_dataset(dataset_folder)
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
