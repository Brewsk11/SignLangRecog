from Classifier.Models import NetworkModels
from hashlib import md5
from time import time
from pickle import dump, load
from os.path import isfile
import keras
from keras.optimizers import *
from keras.callbacks import *

tensors_dir = 'C:/Users/jakub/Desktop/Inzynierka/Tensors/'
models_dir = 'C:/Users/jakub/Desktop/Inzynierka/Models/'
images_tensor_name = 'train_images.tsr'
letters_tensor_name = 'train_letters.tsr'
res = 128
train_validation_ratio = 0.85
test_num = 0
model_class = NetworkModels.ClassifierModel

if __name__ == "__main__":

    #Load image tensor and letters tensor

    print(f"Loading tagged images tensor: {tensors_dir + images_tensor_name}")
    with open(tensors_dir + images_tensor_name, 'rb') as images_file:
        images_tensor = load(images_file)

    print(f"Loading letters tensor: {tensors_dir + letters_tensor_name}")
    with open(tensors_dir + letters_tensor_name, 'rb') as letters_file:
        letters_tensor = load(letters_file)

    #Cut tensors into training and validation

    validation_num = int(images_tensor.length * train_validation_ratio)

    train_images = images_tensor.tensor[:validation_num]
    train_letters = letters_tensor.tensor[:validation_num]

    validation_images = images_tensor.tensor[validation_num:]
    validation_letters = letters_tensor.tensor[validation_num:]


    print('Images tensor shape: ' + str(images_tensor.tensor.shape))
    print('Letters tensor shape: ' + str(letters_tensor.tensor.shape))
    print('First 50 image letters and letters tensors:')
    for x in range(0, 50):
        print(f'Letter: {images_tensor._img_list[x]._letter}, tensor: {letters_tensor._tensor[x]}')


    #Train
    model_hash = md5(str(time()).encode('ascii')).hexdigest()[:5]

    model = model_class().build_model((res, res, 1))
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])

    model.summary()

    epochs_num = 40
    batch_size = 16
    save_every_n_epoch = 4

    history = model.fit(
                    x=train_images,
                    y=train_letters,
                    epochs=epochs_num,
                    batch_size=batch_size,
                    validation_data=(validation_images, validation_letters),
                    callbacks=[
                        ModelCheckpoint(
                            filepath=f'{models_dir}/{model_hash}_{model_class.__name__}_e{{epoch:03d}}',
                            period=save_every_n_epoch)
                    ])

    # Save the final model if it did not save before
    if not isfile(f'{models_dir}/{model_hash}_{model_class.__name__}_e{epochs_num:03d}'):
        print('Saving final model...')
        model.save(filepath=f'{models_dir}/{model_hash}_{model_class.__name__}_e{epochs_num:03d}')

    # Save history
    with open(f'{models_dir}/{model_hash}_{model_class.__name__}_history.p', 'wb') as history_file:
        print('Saving training history...')
        dump(history, history_file)




