#
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
import time
import csv
import matplotlib.pyplot as plt

NAME = "Cruz Oviedo Diego"
OUTPUT_DIR = "p4media"

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
        y.append(x)
        for i in range(len(self.biases)):
            u = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-u))
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
        delta = (y[-1]-t) * y[-1] * (1-y[-1])
        nabla_b [-1] = delta
        nabla_w [-1] = numpy.dot(delta, y[-2].T)  

        for i in range (2, self.num_layers):
            delta = numpy.dot(self.weights[-i+1].T, delta) * y[-i] * (1 - y[-i])
            nabla_b [-i] = delta 
            nabla_w [-i] = numpy.dot(delta, y[-i-1].T)
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
        self.weights = [w-eta*nw for w,nw in zip(self.weights, batch_nabla_w)]
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
        try:
            with open(folder + "data" + str(i), "rb") as f:
                f_data = [c/255.0 for c in f.read(784000)]
        except FileNotFoundError:
             print(f"Error: Data file not found in {folder + 'data' + str(i)}")
             return [], []
             
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])
        training_dataset += images[0:len(images)//2]
        training_labels  += [label for j in range(len(images)//2)]
        testing_dataset  += images[len(images)//2:len(images)]
        testing_labels   += [label for j in range(len(images)//2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))


def run_test_and_evaluate(training_dataset, testing_dataset, epochs, batch_size, learning_rate):
    if not rclpy.ok():
        raise Exception("rclpy is shutdown")
    
    nn = FCNeuralNetwork([784,30,10])
    
    training_start_time = time.time()
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    success_count = 0
    
    for i in range(100):
        img, label = testing_dataset[numpy.random.randint(0,4999)]
        y = nn.feedforward(img)[-1]
        expected_output = numpy.argmax(label)
        recognized_digit = numpy.argmax(y)
        success_count += 1 if expected_output == recognized_digit else 0 
        
    print(f"Test Finished. Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"Time: {training_time:.2f} s. Success: {success_count}/100")
    
    return success_count, training_time

def create_graphs(results_list, output_directory):
    print("\nCreating graphs...")
    
    results_array = numpy.array(results_list)
    
    epochs_unique = numpy.unique(results_array[:, 0])
    batch_unique = numpy.unique(results_array[:, 1])
    lr_unique = numpy.unique(results_array[:, 2])
    
    
    def aggregate_metrics(param_index, unique_params, metric_index):
        aggregated = []
        for unique_val in unique_params:
            filtered_rows = results_array[results_array[:, param_index] == unique_val]
            if filtered_rows.size > 0:
                aggregated.append(numpy.mean(filtered_rows[:, metric_index])) 
            else:
                aggregated.append(0)
        return list(unique_params), aggregated

    epochs_data, success_epochs = aggregate_metrics(0, epochs_unique, 3)
    batch_data, success_batch = aggregate_metrics(1, batch_unique, 3)
    lr_data, success_lr = aggregate_metrics(2, lr_unique, 3)

    _, time_epochs = aggregate_metrics(0, epochs_unique, 4)
    _, time_batch = aggregate_metrics(1, batch_unique, 4)
    _, time_lr = aggregate_metrics(2, lr_unique, 4)

    
    def plot_and_annotate(x_data, y_data, title, x_label, y_label, color, filename):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(x_data, y_data, 'o-', color=color)
        
        
        for x, y in zip(x_data, y_data):
            ax.annotate(f'({x}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            
        plt.savefig(os.path.join(output_directory, filename))
        plt.close(fig)


    # 1. Success vs Epochs
    plot_and_annotate(epochs_data, success_epochs, "Success vs Epochs", "Epochs", "Success", 'blue', "success_vs_epochs.png")

    # 2. Success vs Batch Size
    plot_and_annotate(batch_data, success_batch, "Success vs Batch Size", "Batch size", "Success", 'red', "success_vs_batch_size.png")

    # 3. Success vs Learning Rate
    plot_and_annotate(lr_data, success_lr, "Success vs Learning Rate", "Learning rate", "Success", 'green', "success_vs_learning_rate.png")

    # 4. Time vs Epochs
    plot_and_annotate(epochs_data, time_epochs, "Time vs Epochs", "Epochs", "Training time [s]", 'darkviolet', "time_vs_epochs.png")

    # 5. Time vs Batch Size
    plot_and_annotate(batch_data, time_batch, "Time vs Batch Size", "Batch size", "Training time [s]", 'orange', "time_vs_batch_size.png")
 
    # 6. Time vs Learning Rate
    plot_and_annotate(lr_data, time_lr, "Time vs Learning Rate", "Learning rate", "Training time [s]", 'aqua', "time_vs_learning_rate.png")
    
    print(f"Graphs successfully saved to the '{output_directory}' directory.")

def main(args=None):
    rclpy.init(args=args)
    print("TRAINING AND ANALYZING NEURAL NETWORK - " + NAME)
    
    try:
        package_path = get_package_share_directory("neural_networks")
        dataset_folder = os.path.join(package_path, "dataset")
    except Exception as e:
        print(f"Error: Could not find package 'neural_networks': {e}")
        rclpy.shutdown()
        return

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(script_dir, OUTPUT_DIR)
    
    
    os.makedirs(output_directory, exist_ok=True)


    #PARAMETERS
    epochs_list      = [3, 10, 50, 100]
    batch_sizes    = [5, 10, 30, 100]
    learning_rates = [0.5, 1.0, 3.0, 10.0]
    
    all_results = []
    
    training_dataset, testing_dataset = load_dataset(dataset_folder)
    if not training_dataset:
        rclpy.shutdown()
        return
        
    total_experiments = len(epochs_list) * len(batch_sizes) * len(learning_rates)
    print(f"Starting {total_experiments} experiments...")
    
    for epoch in epochs_list:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                if not rclpy.ok():
                    break
                    
                print(f"\n--- Test: E={epoch}, B={batch_size}, LR={learning_rate} ---")
                
                success_count, training_time = run_test_and_evaluate(
                    training_dataset, testing_dataset, epoch, batch_size, learning_rate
                )
                
                all_results.append([
                    float(epoch), 
                    float(batch_size), 
                    float(learning_rate), 
                    float(success_count), 
                    training_time
                ])
                
        if not rclpy.ok():
            break

    # -SAVE CSV
    CSV_FILENAME = "results_nn.csv"
    CSV_HEADER = ["epochs", "batch_size", "learning_rate", "success_count", "training_time"]
    
    print("\nSaving results to CSV...")
    csv_filepath = os.path.join(output_directory, CSV_FILENAME)
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_results)
        
    print(f"Results saved to {csv_filepath}")
    
    
    create_graphs(all_results, output_directory)
    
    
    print("\nPress ESC to exit.")
    cmd = cv2.waitKey(0)
    while cmd != 27 and rclpy.ok():
        cmd = cv2.waitKey(100)
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
