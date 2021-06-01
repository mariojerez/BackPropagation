'''
By: Mario Jerez
Date: 6 May 2021

Implementing a basic version of the backpropagation learning algorithm
for a neural network of 2 inputs, 2 hidden, and 1 output unit.
Taught a network to mimic simple logical functions such as AND and XOR.
Conducted several experiments to evaluate the effects of momentum and learning rate

run my presentResults() function to see my experimental findings!


-------------------------------------------------------------------
 3-layer neural network with 2 inputs, 1 output, and 2 hidden units

                           output
                             / \
              output_w1 --->/   \<--- output_w2
                           /     \
                       hidden1 hidden2
                          |\     /|
           hidden1_w1 --->| \   / |<--- hidden2_w2
           hidden1_w2 ----|->\ /<-|---- hidden2_w1
                          |   X   |
                          |  / \  |
                        input1 input2

-------------------------------------------------------------------
'''
import random, math, statistics
import PyGnuplot as gp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

class NeuralNetwork:

    def __init__(self):
        # unit activations
        self.output = 0
        self.hidden1 = 0
        self.hidden2 = 0
        # output unit weights
        self.output_bias = random.uniform(-0.1, +0.1)
        self.output_w1 = random.uniform(-0.1, +0.1)
        self.output_w2 = random.uniform(-0.1, +0.1)
        # hidden1 unit weights
        self.hidden1_bias = random.uniform(-0.1, +0.1)
        self.hidden1_w1 = random.uniform(-0.1, +0.1)
        self.hidden1_w2 = random.uniform(-0.1, +0.1)
        # hidden2 unit weights
        self.hidden2_bias = random.uniform(-0.1, +0.1)
        self.hidden2_w1 = random.uniform(-0.1, +0.1)
        self.hidden2_w2 = random.uniform(-0.1, +0.1)
        # learning parameters
        self.tolerance = 0.1
        self.epoch_limit = 50000
        self.learning_rate = 0.1
        self.momentum = 0  # for part 2
        # for adding momentum
        self.output_bias_change = 0
        self.hidden1_bias_change = 0
        self.hidden2_bias_change = 0
        self.output_w1_change = 0
        self.output_w2_change = 0
        self.hidden1_w1_change = 0
        self.hidden1_w2_change = 0
        self.hidden2_w1_change = 0
        self.hidden2_w2_change = 0

    def __str__(self):
        s = "           bias     w1      w2\n"
        s += "output   {:+.4f} {:+.4f} {:+.4f}\n".format(
            self.output_bias, self.output_w1, self.output_w2)
        s += "hidden1  {:+.4f} {:+.4f} {:+.4f}\n".format(
            self.hidden1_bias, self.hidden1_w1, self.hidden1_w2)
        s += "hidden2  {:+.4f} {:+.4f} {:+.4f}".format(
            self.hidden2_bias, self.hidden2_w1, self.hidden2_w2)
        return s
        
    def set_weights(self, weight_list):
        assert type(weight_list) is list and len(weight_list) == 9, \
            "set_weights requires a list of 9 weight/bias values"
        # weights in weight_list must be given in the order below:
        self.output_bias  = weight_list[0]
        self.output_w1    = weight_list[1]
        self.output_w2    = weight_list[2]
        self.hidden1_bias = weight_list[3]
        self.hidden1_w1   = weight_list[4]
        self.hidden1_w2   = weight_list[5]
        self.hidden2_bias = weight_list[6]
        self.hidden2_w1   = weight_list[7]
        self.hidden2_w2   = weight_list[8]

        self.output_bias_change = 0
        self.hidden1_bias_change = 0
        self.hidden2_bias_change = 0
        self.output_w1_change = 0
        self.output_w2_change = 0
        self.hidden1_w1_change = 0
        self.hidden1_w2_change = 0
        self.hidden2_w1_change = 0
        self.hidden2_w2_change = 0

    def initialize(self):
        self.set_weights([random.uniform(-0.1, +0.1) for i in range(9)])
        print("weights randomized")

    def test(self, patterns, targets):
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            if abs(target - output) <= self.tolerance:
                print("{} --> {}".format(pattern, output))
            else:
                print("{} --> {} \t(WRONG, should be {})".format(pattern, output, target))
        error, correct = self.total_error(patterns, targets)
        print("TSS error = {:.5f}, correct = {:.3f}".format(error, correct))

    #--------------------------------------------------------------------------
    # methods to be implemented

    # returns the output produced by the network for the given pattern
    def propagate(self, pattern):
        self.hidden1 = sigmoid(pattern[0] * self.hidden1_w1 + pattern[1] * self.hidden1_w2 + self.hidden1_bias)
        self.hidden2 = sigmoid(pattern[0] * self.hidden2_w1 + pattern[1] * self.hidden2_w2 + self.hidden2_bias)
        self.output = sigmoid(self.hidden1 * self.output_w1 + self.hidden2 * self.output_w2 + self.output_bias)
        return self.output

    # returns a tuple of values (e, c) where e is the total sum squared error
    # for all patterns in the given dataset, and c is the fraction of output
    # values that are within self.tolerance of the given target values
    def total_error(self, patterns, targets):
        TSSError = 0
        numCorrectOutputs = 0
        for p in range(len(patterns)):
            output = self.propagate(patterns[p])
            error = (targets[p] - output)**2
            TSSError += error
            if abs(output - targets[p]) <= self.tolerance:
                numCorrectOutputs += 1
        return TSSError, numCorrectOutputs / len(targets)
                

    # updates network weights and biases for the given pattern and target
    def adjust_weights(self, pattern, target):
        self.propagate(pattern)
        delta_out = (self.output - target) * self.output * (1 - self.output)
        delta_hidden1 = (delta_out * self.output_w1) * self.hidden1 * (1 - self.hidden1)
        delta_hidden2 = (delta_out * self.output_w2) * self.hidden2 * (1 - self.hidden2)
        
        self.output_w1_change = -self.learning_rate * delta_out * self.hidden1 + self.momentum * self.output_w1_change
        self.output_w2_change = -self.learning_rate * delta_out * self.hidden2 + self.momentum * self.output_w2_change
        self.output_bias_change = -self.learning_rate * delta_out + self.momentum * self.output_bias_change

        self.hidden1_w1_change = -self.learning_rate * delta_hidden1 * pattern[0] + self.momentum * self.hidden1_w1_change
        self.hidden1_w2_change = -self.learning_rate * delta_hidden1 * pattern[1] + self.momentum * self.hidden1_w2_change
        self.hidden1_bias_change = -self.learning_rate * delta_hidden1 + self.momentum * self.hidden1_bias_change

        self.hidden2_w1_change = -self.learning_rate * delta_hidden2 * pattern[0] + self.momentum * self.hidden2_w1_change
        self.hidden2_w2_change = -self.learning_rate * delta_hidden2 * pattern[1] + self.momentum * self.hidden2_w2_change
        self.hidden2_bias_change = -self.learning_rate * delta_hidden2 + self.momentum * self.hidden2_bias_change

        self.output_w1 += self.output_w1_change
        self.output_w2 +=  self.output_w2_change
        self.output_bias += self.output_bias_change
        self.hidden1_w1 += self.hidden1_w1_change
        self.hidden1_w2 +=  self.hidden1_w2_change
        self.hidden1_bias +=  self.hidden1_bias_change
        self.hidden2_w1 += self.hidden2_w1_change
        self.hidden2_w2 +=  self.hidden2_w2_change
        self.hidden2_bias += self.hidden2_bias_change


    # trains the network on all of the given patterns and targets
    # until all outputs are within tolerance of the targets
    def train(self, patterns, targets, returnEpoch = False, quiet = False):
        epochNum = 0
        e, c = self.total_error(patterns, targets)
        print("Epoch # \t {}: TSS error = {}, correct = {}".format(epochNum, e, c)) 
        while c != 1 and epochNum < self.epoch_limit:
            for p in range(len(patterns)):
                self.adjust_weights(patterns[p], targets[p])
            e, c = self.total_error(patterns, targets)
            if not quiet:
                print("Epoch # \t {}: TSS error = {}, correct = {}".format(epochNum, e, c))
            epochNum += 1
        if c != 1:
            print("Failed to learn targets within E epochs")
        if returnEpoch:
            return epochNum

## Experimentation

def experiment(learning_rate, momentum, numTrials):
    sumEpochs = 0
    for t in range(numTrials):
        n.learning_rate = learning_rate
        n.momentum = momentum
        n.initialize()
        numEpochs = n.train(inputs, XNORtargets, True, True) #Change to target of interest
        sumEpochs += numEpochs
    avg = sumEpochs / numTrials
    return avg
        
    

def collect_data(learningRate_vals = [0.1, 0.5, 0.7], momentum_vals = [0., 0.5, 0.9], numTrials=5):
    m = [-1] #-1 just means blank space
    for m_val in momentum_vals:
        m.append(m_val)

    matrix = [m]
    for lr_val in learningRate_vals:
        matrix.append([lr_val])
    for r in range(1, len(matrix)):
        for momentum in momentum_vals:
            avgNumEpochs = experiment(matrix[r][0], momentum, numTrials)
            matrix[r].append(avgNumEpochs)
    print(matrix)
    
            
##--------------------------------------------------------------------
ANDmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 4134.4, 2901.6, 2070.2, 1248.2, 414.0], [0.3, 1369.2, 979.8, 686.8, 410.4, 138.0], [0.5, 820.0, 575.2, 410.2, 249.4, 83.0], [0.7, 579.4, 409.4, 290.8, 175.6, 59.0], [0.9, 444.6, 311.6, 224.2, 136.6, 45.4]]
XORmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 30162.4, 14949.8, 10894.6, 6345.6, 2169.6], [0.3, 14266.4, 3268.0, 2629.2, 1577.0, 20296.4], [0.5, 17342.8, 11950.4, 16304.2, 30349.2, 14283.4], [0.7, 12724.0, 5438.6, 20767.0, 11527.0, 20267.4], [0.9, 13957.0, 15517.4, 24309.4, 3831.0, 20149.6]]
ORmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 3390.4, 2381.2, 1711.6, 1024.6, 349.6], [0.3, 1152.6, 812.4, 589.4, 344.2, 118.0], [0.5, 702.4, 473.8, 350.6, 207.0, 71.4], [0.7, 495.0, 336.8, 248.8, 147.2, 51.4], [0.9, 383.4, 273.6, 192.4, 117.2, 39.4]]
NANDmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 4098.8, 2913.4, 2072.6, 1255.6, 421.0], [0.3, 1364.2, 955.6, 685.2, 405.4, 139.6], [0.5, 811.4, 576.2, 409.6, 247.4, 82.0], [0.7, 579.0, 406.8, 291.0, 176.4, 59.2], [0.9, 445.6, 314.4, 226.2, 135.4, 44.8]]
NORmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 3412.0, 2441.2, 1716.8, 1030.0, 348.0], [0.3, 1139.8, 810.0, 563.2, 347.2, 118.6], [0.5, 698.4, 470.2, 344.8, 202.2, 70.8], [0.7, 496.2, 354.2, 249.6, 148.8, 51.6], [0.9, 383.4, 268.2, 190.8, 116.8, 39.0]]
XNORmatrix = [[-1, 0.0, 0.3, 0.5, 0.7, 0.9], [0.1, 26663.6, 13375.6, 10227.6, 5990.0, 2009.6], [0.3, 4761.6, 13672.2, 15927.6, 1441.4, 506.0], [0.5, 21740.4, 11711.2, 30591.8, 10908.0, 354.6], [0.7, 9681.4, 11742.2, 11698.4, 1345.0, 885.6], [0.9, 11706.0, 31041.4, 928.2, 476.4, 18640.8]]

def threeDPlot(matrix):
    learning_rates = [matrix[i][0] for i in range(1, len(matrix))]
    momentums = matrix[0][1:]
    print(learning_rates, momentums)
    ax = plt.axes(projection='3d')
    Y, X = np.meshgrid(learning_rates, momentums)
    Z = np.array([r[1:] for r in matrix[1:]])
    print(Z)
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Momentum")
    ax.set_title("Average Epochs")
    plt.show()
    return X, Y, Z

def printMatrix(m):
    print("Average Epochs (5 trials)")
    print("columns = momentum\nrows = learning rate")
    for r in m:
        row = ""
        for c in r:
            row += "\t" + str(c)
        print(row)

def presentResults():
    input("Press enter to see XOR experimentation results")
    printMatrix(XORmatrix)
    threeDPlot(XORmatrix);
    print("Optimal Settings\nMomentum = 0.7 \t Learning Rate = 0.3")

    input("Press enter to see XNOR experimentation results")
    printMatrix(XNORmatrix)
    threeDPlot(XNORmatrix)
    print("Optimal Settings\nMomentum = 0.9 \t Learning Rate = 0.5")

    input("Press enter to see AND experimentation results")
    printMatrix(ANDmatrix)
    threeDPlot(ANDmatrix)
    print("Optimal Settings\nMomentum = 0.9 \t Learning Rate = 0.9")

    input("Press enter to see OR experimentation results")
    printMatrix(ORmatrix)
    threeDPlot(ORmatrix)
    print("Optimal Settings\nMomentum = 0.9 \t Learning Rate = 0.9")

    input("Press enter to see NAND experimentation results")
    printMatrix(NANDmatrix)
    threeDPlot(NANDmatrix)
    print("Optimal Settings\nMomentum = 0.9 \t Learning Rate = 0.9")

    input("Press enter to see NOR experimentation results")
    printMatrix(NORmatrix)
    threeDPlot(NORmatrix)
    print("Optimal Settings\nMomentum = 0.9 \t Learning Rate = 0.9")

    input("Press enter to see summary of findings")
    print("For all of the linearly separable targets, a high (0.9 or above) momentum and learning rate worked best.")
    print("For the non-linearly separable targets, there was some variation. It seems they prefer a lower learning rate.")
    
    
# learning rate, momentum [0.1, 0.3, 0.5, 0.7, 0.9], [0., 0.3, 0.5, 0.7, 0.9]

#------------------------------------------------------------------------------

inputs = [[0,0], [0,1], [1,0], [1,1]]

ANDtargets  = [0, 0, 0, 1]
ORtargets   = [0, 1, 1, 1]
NANDtargets = [1, 1, 1, 0]
NORtargets  = [1, 0, 0, 0]
XORtargets  = [0, 1, 1, 0]  # not linearly separable
XNORtargets = [1, 0, 0, 1]  # not linearly separable

n = NeuralNetwork()

