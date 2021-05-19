from __future__ import division
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
import keras
from datetime import datetime

max_epochs = 5
k = -0.2


class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    no_parameters = np.sum(mask_weights)
    print("Create Sparse Matrix: NoParameters, NoRows, NoCols ", no_parameters, noRows, noCols)
    return [no_parameters, mask_weights]


class AccSET_MLP_CIFAR_10:
    def __init__(self):
        self.number_of_connections_per_epoch = 0
        self.current_accuracy = 0

        # set model parameters
        self.epsilon = 20  # control the sparsity level as discussed in the paper
        self.zeta = 0.3  # the fraction of the weights removed
        self.batch_size = 100  # batch size
        self.maxepoches = max_epochs  # number of epochs
        self.learning_rate = 0.01  # SGD learning rate
        self.num_classes = 10  # number of classes
        self.momentum = 0.9  # SGD momentum

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, 1000, 4000)

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        # initialize weights for LReLu activation function
        self.wLRelu1 = None
        self.wLRelu2 = None
        self.wLRelu3 = None

        # saved weights after each batch
        self.weightsPerBatch = None

        self.create_model()

        self.train()

    def create_model(self):
        # create a model for CIFAR10 with 3 hidden layers
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(32, 32, 3)))
        self.model.add(Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1), weights=self.w1))
        self.model.add(keras.layers.LeakyReLU(name="lrelu1", weights=self.wLRelu1))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2), weights=self.w2))
        self.model.add(keras.layers.LeakyReLU(name="lrelu2", weights=self.wLRelu2))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3), weights=self.w3))
        self.model.add(keras.layers.LeakyReLU(name="lrelu3", weights=self.wLRelu3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, name="dense_4",
                             weights=self.w4))
        self.model.add(Activation('softmax'))
        self.optim = keras.optimizers.SGD(lr=0.01, momentum=0.975, decay=2e-06, nesterov=True)

    def read_data(self):
        # read CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # normalize data
        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTtrainStd
        x_test = (x_test - xTrainMean) / xTtrainStd

        return [x_train, x_test, y_train, y_test]


    def train(self):
        training_start = datetime.now().timestamp()
        # read CIAFAR10 data
        [x_train, x_test, y_train, y_test] = self.read_data()
        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        self.model.summary()

        # training process in a for loop
        for epoch in range(0, self.maxepoches):
            self.number_of_connections_per_epoch = 0
            sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                                batch_size=self.batch_size),
                                                   steps_per_epoch=x_train.shape[0] // self.batch_size,
                                                   epochs=epoch,
                                                   validation_data=(x_test, y_test),
                                                   initial_epoch=epoch - 1)

            self.current_accuracy = historytemp.history['val_acc'][0]

            self.weightsEvolution()
            K.clear_session()
            self.create_model()


    def weightsEvolution(self):
        # Remove the weights closest to zero in each layer and add new random weights
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wLRelu1 = self.model.get_layer("lrelu1").get_weights()
        self.wLRelu2 = self.model.get_layer("lrelu2").get_weights()
        self.wLRelu3 = self.model.get_layer("lrelu3").get_weights()

        [self.wm1, self.wm1Core] = self.rewireMask(self.w1[0], self.noPar1)
        [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], self.noPar2)
        [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], self.noPar3)

        self.w1[0] = self.w1[0] * self.wm1Core
        self.w2[0] = self.w2[0] * self.wm2Core
        self.w3[0] = self.w3[0] * self.wm3Core

    def rewireMask(self, weights, no_weights):
        # rewire weight matrix
        # remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)
        largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
         Kuhli loachsmallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]
        rewired_weights = weights.copy()
        rewired_weights[rewired_weights > smallest_positive] = 1
        rewired_weights[rewired_weights < largest_negative] = 1
        rewired_weights[rewired_weights != 1] = 0
        weight_mask_core = rewired_weights.copy()

        # add zeta random weights
        nr_add = 0
        # Number of connections to be added
        no_rewires = self.calculate_number_of_connections_to_add(rewired_weights, no_weights)
        while nr_add < no_rewires:
            i = np.random.randint(0, rewired_weights.shape[0])
            j = np.random.randint(0, rewired_weights.shape[1])
            if rewired_weights[i, j] == 0:
                rewired_weights[i, j] = 1
                nr_add += 1
        self.number_of_connections_per_epoch = self.number_of_connections_per_epoch + np.sum(rewired_weights)

        return [rewired_weights, weight_mask_core]

    def calculate_number_of_connections_to_add(self, rewired_weights, no_weights):
        # Adjust sparsity based on the accuracy
        x = self.current_accuracy
        y = 1 - ((x - x * k) / (k - abs(x) * 2 * k + 1))
        return (no_weights - np.sum(rewired_weights)) * y


if __name__ == '__main__':
    # create and run a AccSET-MLP model on CIFAR10
    model = AccSET_MLP_CIFAR_10()
