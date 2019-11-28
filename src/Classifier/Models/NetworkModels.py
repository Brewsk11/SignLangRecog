from keras.models import Model as KerasModel, Sequential
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, concatenate, \
    BatchNormalization, Activation, ZeroPadding2D

classes_number = 27

class ClassifierModel:

    def build_model(self, input_shape: tuple) -> KerasModel:

        input = Input(shape=input_shape)

        # convolutional layer 1
        conv1 = self.convolutional_block(input, 10, (5, 5))
        pool1 = MaxPooling2D((2, 2))(conv1)
        dropout1 = Dropout(0.5)(pool1)

        # convolutional layer 2
        padding = ZeroPadding2D((1, 1))(dropout1)
        conv2 = self.convolutional_block(padding, 20, (3, 3))
        pool2 = MaxPooling2D(2, 2)(conv2)
        dropout2 = Dropout(0.5)(pool2)

        flat = Flatten()(dropout2)

        dense = Dense(classes_number, activation='softmax')(flat)

        model = KerasModel(input=input, output=dense)

        return model

    def convolutional_block(self, input, channels, kernel_size):
        conv = Conv2D(channels, kernel_size)(input)
        batch_norm = BatchNormalization(axis=3)(conv)
        relu = Activation('relu')(batch_norm)
        return relu

class ClassifierModel2:

    def build_model(self, input_shape: tuple) -> KerasModel:
        input = Input(shape=input_shape)
        conv1 = Conv2D(16, 1)(input)
        conv2 = Conv2D(16 ,1)(conv1)
        maxpool1 = MaxPooling2D(2, 2)(conv2)
        conv3 = Conv2D(32, 1)(maxpool1)
        conv4 = Conv2D(32, 1)(conv3)
        maxpool2 = MaxPooling2D(2, 2)(conv4)
        conv5 = Conv2D(64, 1)(maxpool2)
        conv6 = Conv2D(64, 1)(conv5)
        maxpool3 = MaxPooling2D(2, 2)(conv6)
        conv7 = Conv2D(128, 1)(maxpool3)
        flat = Flatten()(conv7)
        dense = Dense(classes_number, activation='softmax')(flat)

        model = KerasModel(input=input, output=dense)

        return model