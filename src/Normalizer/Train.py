from keras.optimizers import *
from keras.callbacks import *
from pickle import dump, load
from hashlib import md5
from time import time
from os.path import isfile

from Normalizer.Models import NetworkModels


resources_dir = '/home/pawel/PracaInzynierska/Normalizer'

tensors_dir = resources_dir + '/Tensors'
models_dir = resources_dir + '/Models'

model_class = NetworkModels.UNetModel


if __name__ == "__main__":

    # 1: -- Load images and convert to tensors --

    in_res = 128
    out_res = 128

    training_data_tensor = tensors_dir + f'/training_tensor_{str(in_res)}.tsr'
    tagged_data_tensor = tensors_dir + f'/tagged_tensor_{str(out_res)}.tsr'

    print(f'Loading tensors:')
    print(f'Training data: {training_data_tensor}')
    print(f'Tagged data:   {tagged_data_tensor}')

    with open(training_data_tensor, 'rb') as training_file, open(tagged_data_tensor, 'rb') as tagged_file:

        org_tsr = load(training_file)
        tgg_tsr = load(tagged_file)

    # 2: -- Cut the tensor into training, validation and test sets --

    # Check if both tensors are the same length
    if org_tsr.length != tgg_tsr.length:
        raise RuntimeError(f'Loaded tensors {training_data_tensor} and {tagged_data_tensor} and different lengths: '
                           f'{str(org_tsr.length)} vs {str(tgg_tsr.length)}')

    samples_num = org_tsr.length
    test_num = 30  # 30 samples to test the network
    train_num = int((samples_num - test_num) * 0.9)  # 90% of samples will be used for training
    valid_num = samples_num - (test_num + train_num)  # The rest we will use for network validation

    org_train = org_tsr.tensor[:train_num]
    tgg_train = tgg_tsr.tensor[:train_num]

    org_valid = org_tsr.tensor[train_num:(train_num + valid_num)]
    tgg_valid = tgg_tsr.tensor[train_num:(train_num + valid_num)]

    org_test = org_tsr.tensor[(train_num + valid_num):]
    tgg_test = tgg_tsr.tensor[(train_num + valid_num):]

    # 3: -- Train the network --

    model_hash = md5(str(time()).encode('ascii')).hexdigest()[:5]

    model = model_class().build_model((in_res, in_res, 1), (out_res, out_res, 1))
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.summary()

    epochs_num = 3
    batch_size = 32
    save_every_n_epoch = 1  # How often the fit function will save the model to the models directory

    history = model.fit(
                    x=org_train,
                    y=tgg_train,
                    epochs=epochs_num,
                    batch_size=batch_size,
                    validation_data=(org_valid, tgg_valid),
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
