from keras.models import Model as KerasModel, Sequential
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, concatenate, \
    BatchNormalization, Activation, ZeroPadding2D

classes_number = 25

class ClassifierModel:

    def build_model(self, input_shape: tuple) -> KerasModel:

        input = Input(shape=input_shape)

        #convolutional layer 1
        conv1 = self.convolutional_block(input, 10, (5,5))
        pool1 = MaxPooling2D((2, 2))(conv1)
        dropout1 = Dropout(0.5)(pool1)

        #convolutional layer 2
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
        batch_norm = BatchNormalization(axis = 3)(conv)
        relu = Activation('relu')(batch_norm)
        return relu
