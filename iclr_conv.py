# import modules
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.initializers import *
from keras import backend as K

K.set_image_dim_ordering('th')


# Preprocess data

# Set random seed for reproducibility
seed = 0
numpy.random.seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

# mean variance
def variance(X_test):
    """Function to calculate variance"""
    sum_of_all = 0.0
    for image in X_test:
        N = len(X_test)
        for container in image:
            sum_of_datapoint = 0.0
            for row in container:
                for value in row:
                    sum_of_datapoint += value
        avg = sum_of_datapoint/784
        sum_of_all += avg**2
    return sum_of_all/N

constant_variance_testing = variance(X_test)


# Define class with neural network
class ConvNet:
    def __init__(self, init_type, init_scale, init_seed):
        self.init_type = init_type      # parameter for init type
        self.init_seed = init_seed      # parameter for seed (should be None)
        self.init_scale = init_scale    # parameter for scale

        # metrics
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

    def run_conv_net(self, X_train, X_test, y_train, y_test, epoch_num):
        """Run model and retain information"""

        # build model
        model = Sequential()

        model.add(Conv2D(16, (5, 5), input_shape=(1, 28, 28), activation='relu', kernel_initializer=glorot_uniform(seed=self.init_seed)))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

        # fit model
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=200, verbose=2)

        # retain information
        self.loss = hist.history['loss']
        self.val_loss = hist.history['val_loss']
        self.acc = hist.history['acc']
        self.val_acc = hist.history['val_acc']

    def return_history(self):
        """Returns history in following format:
        [loss, val_loss, acc, val_acc]"""
        return_list = [self.loss, self.val_loss, self.acc, self.val_acc]
        return return_list
