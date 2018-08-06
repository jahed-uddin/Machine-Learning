import math
import pickle
import numpy as np
from PIL import Image


def sigmoid(x):
    # Apply Sigmoid function to Ints, Floats, lists, and np.arrays
    if type(x) in [int, float]:
        return 1.0 / (1 + math.exp(-x))
    if type(x) == list:
        new_theta = [[(1.0 / (1 + math.exp(-x[column][i])))
                      for i in range(len(x[column]))] for column in range(len(x))]
        return new_theta
    if type(x) == np.ndarray:
        return 1.0 / (1 + np.exp(-x.copy()))
    else:
        raise TypeError("Incorrect Data Type")


def sigmoid_gradient(x):
    # Apply Sigmoid function to Ints, Floats, lists, and np.arrays
    if type(x) in [int, float]:
        return sigmoid(x) * (1.0 - sigmoid(x))
    if type(x) == list:
        sig_x = sigmoid(x)   # 1st expression
        expr_2 = sigmoid(x)
        expr_2 = [[(1.0 - v) for v in line] for line in expr_2]
        new_theta = [[sig_x[r][c] * expr_2[r][c] for c in range(len(x[r]))] for r in range(len(x))]
        return new_theta
    if type(x) == np.ndarray:
        sig_x = sigmoid(x)
        return sig_x * (1 - sig_x)
    else:
        raise TypeError("Incorrect Data Type")


# Deprecated, O(N) far too exponential, super inefficient, numpy.array will have to be used after all
def mult_matrices(matrix1, matrix2):
    # m1 = [[1, 2, 3], [4, 5, 6]]
    # m2 = [[7,8],[9,10],[11,12]]
    dimension_1 = len(matrix1)
    dimension_2 = len(matrix2[1])
    new_matrix = [[0 for _ in range(dimension_2)] for _ in range(dimension_1)]

    for outer_loop in range(len(new_matrix)):
        for inner_loop in range(len(new_matrix[outer_loop])):
            m1_vals = [i for i in matrix1[outer_loop]]
            m2_vals = [i[inner_loop] for i in matrix2]
            value = sum([k * v for k, v in zip(m1_vals, m2_vals)])
            new_matrix[outer_loop][inner_loop] = value
    return new_matrix


# Function to convert Y values into arrays
def convert_y_vals(y):
    y = [int(val) for val in y]
    y_matrix = []
    for y_val in y:
        y_mappings = [0 for _ in range(10)]
        if y_val == 10:
            y_mappings[0] = 1.0
        else:
            y_mappings[y_val] = 1.0
        y_matrix.append(y_mappings)
    return y_matrix

# Function to load data
def load_matlab_data():

    # Matlab/Coursera supplied data
    '''
    with open('digit_vals.txt', 'rb') as fh:
        x_vals = pickle.load(fh)
    '''

    # Rotated/reflected data
    x_vals = np.genfromtxt('new_x_vals.csv', delimiter=',')

    with open('digits.txt', 'rb') as fh:
        y = pickle.load(fh)

    x_vals = np.array([list(entry) for entry in x_vals])
    y_vals = np.array(convert_y_vals(y))
    return [x_vals, y_vals]


class NeuralNet(object):
    def __init__(self, i_l_size, h_l_dim, o_l_size, **kwargs):
        # nn = NeuralNet(2, [2, 2], 2)
        # i_l_size = input layer size
        # h_l_dim = hidden_layer dimensions
        # o_l_size = output layer size
        self.il_size = i_l_size #
        self.hl_dim = h_l_dim
        self.ol_size = o_l_size

        # units per hidden layer
        self.h_units = h_l_dim[0]

        # num of hidden layers
        self.h_layers = h_l_dim[1]

        # num of parameter tables
        self.thetas = h_l_dim[1] + 1

        # actual theta tables
        self.theta_vals = [np.array([]) for _ in range(self.thetas)]

        # lambda
        self.lda = float(kwargs['lda']) if 'lda' in kwargs else 1.0

        if 'load_data' in kwargs and kwargs['load_data']==True:

            data = load_matlab_data()
            # training data
            self.train_data = np.array(kwargs['train_data']) if 'train_data' in kwargs else np.array(data[0])
            # Must provide train data for train_y to be accepted
            self.train_y = np.array(kwargs['train_y']) if 'train_data' and 'train_y' in kwargs else np.array(data[1])

        # Optimization method: Gradient Descent or Optimization lib
        self.optim_m = kwargs['optim_m'] if 'optim_m' in kwargs \
                        and kwargs['optim_m'] in ['grad_descent', 'fminunc', 'fminc'] else 'grad_descent'

        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0

    def init_thetas(self):
        for i in range(self.thetas):
            # Input Layer
            if i == 0:
                epsilon_init = math.sqrt(6) / math.sqrt(self.il_size + self.h_layers)
                theta = np.random.random((self.h_units, self.il_size + 1)) * (2 * epsilon_init) - epsilon_init
                self.theta_vals[i] = theta
            # Output Layer
            elif i == (self.thetas - 1):
                epsilon_init = math.sqrt(6) / math.sqrt(self.ol_size + self.h_layers)
                theta = np.random.random((self.ol_size, self.h_units + 1)) * (2 * epsilon_init) - epsilon_init
                self.theta_vals[i] = theta
            # Hidden Layer
            else:
                epsilon_init = math.sqrt(6) / math.sqrt(2 * self.h_layers)
                theta = np.random.random((self.h_units, self.h_units + 1)) * (2 * epsilon_init) - epsilon_init
                self.theta_vals[i] = theta

    def compute_cost(self, h_x, train_y, lda=None):
        if not lda: lda = self.lda
        cost = sum(sum((-train_y * np.log(h_x)) - ((1 - train_y) * np.log(1 - h_x))))
        m = len(train_y)
        J = cost / m

        # Regularization
        reg_val = (lda/(2*m)) * sum([sum(sum(theta[:, 1:].copy()**2)) for theta in self.theta_vals])
        J += reg_val
        return J

    def fwd_prop(self, **kwargs):
        # returns [activations, z_vals]
        # Load training data as Numpy object
        train_data = kwargs['train_data'] if 'train_data' in kwargs else self.train_data
        train_data = np.array(train_data)

        activations = [[] for _ in range(self.thetas)]
        z_vals = [[] for _ in range(self.thetas)]
        activation = train_data

        for theta in range(len(self.theta_vals)):
            activations[theta] = activation
            # insert bias value to activations
            activation = np.insert(activation, 0, values=1.0, axis=1)
            z = np.dot(activation, self.theta_vals[theta].transpose())
            z_vals[theta] = z
            # g(z)
            activation = 1.0/(1.0 + np.exp(-z))

        activations.append(activation)
        return [activations, z_vals]

    def back_prop(self, **kwargs):
        lda = kwargs['lda'] if 'lda' in kwargs else self.lda
        self.lda = lda
        m = float(len(self.train_y))
        # Initialize Big Deltas to zero

        Deltas = [np.zeros(theta.shape) for theta in self.theta_vals]
        gradients = [[] for _ in self.theta_vals]

        for i in range(len(self.train_data)):
            activations, z_vals = self.fwd_prop(train_data=[self.train_data[i]])
            h_x = activations[-1]

            deltas = [[] for _ in self.theta_vals]

            for delta in range(len(deltas)-1, -1, -1):
                if delta == (len(deltas)-1):
                    deltas[delta] = (h_x - self.train_y[i]).transpose()
                # iterate backwards,
                else:
                    expr1 = np.dot(self.theta_vals[delta+1].transpose(), deltas[delta+1])
                    expr2 = np.insert(sigmoid_gradient(activations[delta+1]), 0, values=1.0, axis=1)
                    delta_l = (expr1 * expr2.transpose())[1:]
                    deltas[delta] = delta_l

                activ_bias = np.insert(activations[delta], 0, values=1.0, axis=1)
                delta_expr2 = np.dot(deltas[delta], activ_bias)
                Deltas[delta] = Deltas[delta] + delta_expr2

        for i in range(len(Deltas)):
            theta_expr = np.insert(self.theta_vals[i][:,1:], 0, values=0.0, axis=1)
            gradients[i] = ((1/m) * Deltas[i]) + ((lda/m) * theta_expr)

        return gradients

    def grad_descent(self, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        grads = kwargs['theta_grads']

        for theta in range(len(self.theta_vals)):
            new_theta = self.theta_vals[theta] - (alpha * grads[theta])
            self.theta_vals[theta] = new_theta

    def fminunc(self, alpha = 1.0, **kwargs) : # cost, theta_grads, alpha = 1.0):
        pass

    def fminc(self):
        pass

    def optimize(self, theta_grads, **kwargs):
        optim_m = self.optim_m if 'optim_m' not in kwargs else kwargs['optim_m']
        optim = {'grad_descent': self.grad_descent, 'fminunc': self.fminunc, 'fminc': self.fminc}[optim_m]
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha

        # Will calculate and assign new Theta's to self.theta_vals
        optim(theta_grads=theta_grads, alpha=alpha)

    def train(self, **kwargs):

        train_data = kwargs['train_data'] if 'train_data' in kwargs else self.train_data
        train_data = np.array(train_data)
        train_y = kwargs['train_y'] if 'train_y' in kwargs else self.train_y
        train_y = np.array(train_y)
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        self.alpha = alpha

        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 5

        for iteration in range(max_iter):
            activations, z_vals = self.fwd_prop(train_data=train_data)
            h_x = activations[-1]
            cost = self.compute_cost(h_x, train_y)
            theta_grads = self.back_prop()
            print("Iteration {0}: Cost = {1}".format(iteration+1, cost))
            print("Optimizing...")
            self.optimize(theta_grads, alpha=alpha)

    def predict(self, image):
        img = Image.open(image).convert('L')
        img = img.resize((20, 20), Image.ANTIALIAS)
        img_as_np = np.asarray(img)
        img_as_np = img_as_np.reshape(1,400)
        x_vals = np.array(img_as_np)
        activations, z_vals = self.fwd_prop(train_data=x_vals)
        output = activations[-1][0]
        print(output)
        print([1 if i == output.argmax() else 0 for i in range(len(output))])
        print(output.argmax(axis=0))

if __name__ == "__main__":

    # Build NN
    print("Building Artificial Neural Net (ANNet) For Numeric Character Recognition...")
    nn = NeuralNet(400, [10, 2], 10, load_data=True)#load_data=True)


    '''
    # Use learned instances
    print("Loading Cached Learning Parameters...")
    nn.theta_vals[0] = np.genfromtxt('theta-2-0.csv', delimiter=',')
    nn.theta_vals[1] = np.genfromtxt('theta-2-1.csv', delimiter=',')
    '''


    # Assign regularization parameter (Default 1.0)
    nn.lda = 0.3

    # Random Theta initialization
    print("Randomizing Initial Theta values...")
    nn.init_thetas()

    # Print Artificial Neural Net info:
    print('ANNet Input Layer Size: ', nn.il_size)
    print('ANNet Hidden Layer Dimensions: ', nn.hl_dim)
    print('ANNet Output Layer Size: ', nn.ol_size)

    # ----------------------------------

    # Train ANNet
    print('\n' + '=' * 100 + '\nTraining Network...')

    # Set optimization method
    print("Setting Optimization Method...")
    nn.optim_m = 'grad_descent'
    print("Using Gradient Descent Optimization Algorithm...")

    nn.train(max_iter=200, alpha=5.0)
    print("Neural Net Trained!")

    '''

    # Save Learning (Theta) Parameters
    print("Caching Learning Parameters..")
    for thetas in range(len(nn.theta_vals)):
        np.savetxt("theta-2-" + str(thetas) + ".csv", nn.theta_vals[thetas], delimiter=",")
    print("Completed.")

    '''

    print('\n' + '=' * 100 + '\n')
    # ----------------------------------


    # Predict
    print("Calculating Hypothesis...")
    nn.predict("number2.png")
    exit()

    '''
    x_vals = np.array([np.genfromtxt('6_matrix.csv', delimiter=',')])
    activations, z_vals = nn.fwd_prop(train_data=x_vals)
    output = activations[-1][0]
    print(output)
    print([1 if i == output.argmax(axis=0) else 0 for i in range(len(output))])
    print(output.argmax(axis=0))
    '''

######  NOTES   #######

# Forward propagation (feed forward) through the Network
# train_data     = [ [... x400...],     Theta1_dimension = [ [...x400...],
#                    [... x400...],                          [...x400...],
#                    [     ...   ],                          [   ...    ],
#                    [     ...   ],                          [   ...    ],
#                         x5000                                  x25
#                    [     ...   ],                          [   ...    ],


