from pickle import dump, load
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.utils.multiclass import unique_labels

tensors_dir = 'C:/Users/jakub/Desktop/Inzynierka/Tensors/'
models_dir = 'C:/Users/jakub/Desktop/Inzynierka/Models/'
images_tensor_name = 'all_images.tsr'
letters_tensor_name = 'all_letters.tsr'
res = 128
train_validation_ratio = 0.9
test_num = 100

history_file = 'b2404_ClassifierModel_history.p'
model_file = 'b2404_ClassifierModel_e040'
model_path = models_dir + model_file

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized")
    else:
        print("Not normalized")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            pass
            #plt.text(j, i, "{:,}".format(cm[i, j]),
             #        horizontalalignment="center",
              #       color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.grid()
    plt.show()

if __name__ == "__main__":

    with open(f'{models_dir}{history_file}', 'rb') as history_file:
        history = load(history_file)

    with open(f'{models_dir}{model_file}', 'rb') as model_file:
        model = keras.models.load_model(model_path)

    '''
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
    '''
    with open(tensors_dir + 'test_images.tsr', 'rb') as images_file:
        test_images = load(images_file)

    with open(tensors_dir + 'test_letters.tsr', 'rb') as letters_file:
        test_letters = load(letters_file)

    #prediction and create confusion matrix

    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    predicted_letters = model.predict(test_images)
    predicted_letters = np.argmax(predicted_letters, axis=1)
    test_letters = np.argmax(test_letters, axis=1)

    cm = confusion_matrix(test_letters, predicted_letters)
    plot_confusion_matrix(cm, classes, title="Confusion matrix")
    plt.show()

    print("Trt")












