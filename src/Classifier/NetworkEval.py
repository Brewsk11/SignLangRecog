from pickle import dump, load
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import seaborn as sns

tensors_dir = 'C:/Users/jakub/Desktop/Inżynierka/Tensors/'
models_dir = 'C:/Users/jakub/Desktop/Inżynierka/Models/'
images_tensor_name = 'images.tsr'
letters_tensor_name = 'letters.tsr'
res = 128
train_validation_ratio = 0.9
test_num = 100

history_file = '71491_ClassifierModel3_history.p'
model_file = '0bd62_ClassifierModel2_e020'
model_path = models_dir + model_file


if __name__ == "__main__":

    with open(f'{models_dir}{history_file}', 'rb') as history_file:
        history = load(history_file)

    with open(f'{models_dir}{model_file}', 'rb') as model_file:
        model = keras.models.load_model(model_path)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    print(f"Loading tagged images tensor: {tensors_dir + images_tensor_name}")
    with open(tensors_dir + images_tensor_name, 'rb') as images_file:
        images_tensor = load(images_file)

    print(f"Loading letters tensor: {tensors_dir + letters_tensor_name}")
    with open(tensors_dir + letters_tensor_name, 'rb') as letters_file:
        letters_tensor = load(letters_file)

    #Cut tensors into training, validation and test sets

    train_num = int(images_tensor.length * train_validation_ratio)
    validation_num = images_tensor.length - test_num - train_num

    train_images = images_tensor.tensor[:train_num]
    train_letters = letters_tensor.tensor[:train_num]

    validation_images = images_tensor.tensor[train_num:(train_num + validation_num)]
    validation_letters = letters_tensor.tensor[train_num:(train_num + validation_num)]

    test_images = images_tensor.tensor[(train_num + validation_num):]
    test_letters = letters_tensor.tensor[(train_num + validation_num):]

    #prediction and create confusion matrix

    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O',
               'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_letters = model.predict(test_images)


    print("Trt")







