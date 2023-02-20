import numpy as np
from scipy import signal
from matplotlib.image import imread
import cv2












def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


class Layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward_propagation(self, x):
        # TODO: return output
        pass

    def backward_propagation(self, y_gradient, alpha):
        # TODO: update parameters and return input gradient
        pass
    


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, x):
        self.x = x
        return self.activation(self.x)

    def backward_propagation(self, y_gradient, alpha):
        return np.multiply(y_gradient, self.activation_prime(self.x))
    
    
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class sig(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward_propagation(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class C_L(Layer):
    
    
    
    def __init__(self, x_shape, k_size, d):
        x_d, x_h, x_w = x_shape
        self.x_shape = x_shape
        self.x_d = x_d
        self.d = d
        self.k_shape = (d, x_d, k_size, k_size)
        self.y_shape = (d, x_h - k_size + 1, x_w - k_size + 1)
        self.k = np.random.randn(*self.k_shape)
        self.b = np.random.randn(*self.y_shape)
        
        
        
        
    def forward_propagation(self, x):
        self.x = x
        self.y = np.copy(self.b)
        for c in range(self.d):
            for d in range(self.x_d):
                self.y[c] += signal.correlate2d(self.x[d], self.k[c, d], "valid")
        return self.y

    def backward_propagation(self, y_gradient, alpha):
        k_gradient = np.zeros(self.kernels_shape)
        x_gradient = np.zeros(self.input_shape)

        for a in range(self.depth):
            for b in range(self.input_depth):
                k_gradient[a, b] = signal.correlate2d(self.x[b], y_gradient[a], "valid")
                x_gradient[b] += signal.convolve2d(y_gradient[a], self.k[a, b], "full")

        self.k -= alpha * k_gradient
        self.b -= alpha * y_gradient
        return x_gradient

class dense_layer(Layer):
    def __init__(self, x_size, y_size):
        self.w = np.random.randn(y_size, x_size)
        self.b = np.random.randn(y_size, 1)

    def forward_propagation(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def backward_propagation(self, y_gradient, alpha):
        w_gradient = np.dot(y_gradient, self.x.T)
        x_gradient = np.dot(self.w.T, y_gradient)
        self.w -= alpha * w_gradient
        self.b -= alpha * y_gradient
        return 
    
class Reshape_f_b(Layer):
    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self._shapey = y_shape

    def forward_reshape(self, x):
        return np.reshape(x, self.y_shape)

    def backward_reshape(self, y_gradient, alpha):
        return np.reshape(y_gradient, self.x_shape)
    
    
    
list_of_pics_yes = list()
for i in range(0,1500):
    train_image_yes= imread('y'+str(i)+'.jpg')
    train_y = cv2.resize(train_image_yes, (25,25))
    if len(train_y.shape) == 3:
        train_y = cv2.cvtColor(train_y, cv2.COLOR_BGR2GRAY)
    train_y = np.reshape(train_y,(1,25,25))

    train_y = np.append(train_y,1)
    list_of_pics_yes.append(train_y)
list_of_pics_yes = np.asarray(list_of_pics_yes)


# no dataset labeling(0)
list_of_pics_no = list()
for i in range(0,1500):
    train_image_no= imread('no'+str(i)+'.jpg')
    train_n = cv2.resize(train_image_no, (25,25))
    if len(train_n.shape) == 3:
        train_n = cv2.cvtColor(train_n, cv2.COLOR_BGR2GRAY)
    train_n = np.reshape(train_n,(1,25,25))

    train_n = np.append(train_n,0)
    list_of_pics_no.append(train_n)
list_of_pics_no = np.asarray(list_of_pics_no)


# k-fold = 5-fold


a= [0,300,600,900,1200]
b = [300,600,900,1200,1500]

# for l,p in zip(a,b):
    
# y_test dataset labeling(-1)

list_of_pics_ytest = list_of_pics_yes[0:300]
# # no_test labeling(-1)
list_of_pics_ntest = list_of_pics_no[0:300]
test_dataset = np.concatenate((list_of_pics_ntest,list_of_pics_ytest),axis = 0)
x_test = test_dataset[:,:-1]
x_test = x_test.astype("float32")/255

# train dataset and labels dataset


train_dataset = np.concatenate((list_of_pics_no,list_of_pics_yes),axis = 0)
x_train = train_dataset[:,:-1]
x_train = x_train.astype("float32")/255
print(x_train.shape)
# one hot encoding

x_labels = train_dataset[:,-1]
y_labels = test_dataset[:,-1]

train_labels_coded = []
test_abels_coded = []
c1 = np.array([0,1]).T
c2 = np.array([1,0]).T
for i in x_labels:
    if i == 0:
        train_labels_coded.append(c2)
    else:
        train_labels_coded.append(c1)

for i in y_labels:
    if i == 0:
        test_abels_coded.append(c2)
    else:
        test_abels_coded.append(c1)
        
test_abels_coded = np.asarray(test_abels_coded)
train_labels_coded = np.asarray(test_abels_coded)



network = [C_L((1,25,25),3,5), sig(),
           Reshape_f_b((5,23,23),(5*23*23,1)),
           dense_layer(5*23*23,100),
           sig(),
           dense_layer(100,2),
           sig()]

epochs = 20
alpha = 0.1

for t in range(epochs):
    error = 0
    for x,y in zip(x_train,train_labels_coded):
        x =  np.reshape(x,(25,25))
        y_o = x
        for l in network:
            
            y_o = l.forward_propagation(y_o)
        
        error += binary_cross_entropy(y, y_o)
        
        
        g = binary_cross_entropy_prime(y, y_o)
        
        for l in reversed(network):
            g = l.backward(g, alpha)
    
    error/= len(x_train)
    print(f"{t+1}/{epochs}, error = {error}")
    
    
    
for x,y in zip(x_test,test_abels_coded):
    x = np.reshape(x,(25,25))
    y_o = x
    for l in network:
        y_o = l.forward_propagation(y_o)
    print(f"pred : {np.argmax(y_o)} , true : {np.argmax(y)}")
    
    
    
