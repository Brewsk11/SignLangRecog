from keras.models import Model as KerasModel, Sequential
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, concatenate


class NetworkModel:

    def build_model(self, input_shape: tuple, output_shape: tuple) -> KerasModel:
        """
        :param input_shape: Shape of the input tensor
        :param output_shape: Shape of the output tensor
        :return: Keras model object
        """


class UNetModel(NetworkModel):

    def build_model(self, input_shape: tuple, output_shape: tuple) -> KerasModel:
        """
        Prepares convolutional U-Net model sourced from here:
        https://github.com/zhixuhao/unet/blob/master/model.py

        :param input_shape: Should be a (width, height, channels) where width == height, width is a power of 2, and preferably
        no less than 128
        :param output_shape: Must be the same as the input_shape
        :raises RuntimeError: In case the input_shape and output_shape are different
        :return: Keras model object
        """

        if input_shape != output_shape:
            raise RuntimeError(f'Input and output shapes are different: {input_shape} vs. {output_shape}')

        inputs = Input(shape=input_shape)

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = KerasModel(input=inputs, output=conv10)

        return model


class SimpleDenseModel:

    def build_model(self, input_shape: tuple, output_shape: tuple) -> KerasModel:
        """
        Prepares a simple dense network model with just a few dense layers.

        :param input_shape: A (lenght,) tuple
        :param output_shape: A (lenght,) tuple
        :return: Keras model object
        """

        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(output_shape[0] * output_shape[1] * output_shape[2], activation='sigmoid'))
        model.add(Reshape(target_shape=output_shape))

        return model
