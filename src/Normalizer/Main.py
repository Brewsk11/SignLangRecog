import keras
from keras.models import Model
from keras.optimizers import *
from keras.callbacks import *
from pickle import dump, load
from hashlib import md5
from time import time
from os.path import isfile

from Normalizer.Providers.DirectoryImageProvider import DirectoryImageProvider
from Normalizer.Models.TensorBuilder import TensorBuilder
from Normalizer.Models.ImageModels import TrainingImage, TaggedImage
from Normalizer.Models.NetworkModels import UNetModel, SimpleDenseModel


# You should use pickle to dump the loaded tensor as the preparation takes a long time.
# However for the first time or if you change data you can reload images using the flag below.
reload_images = False

training_data_dir = '/home/pawel/PracaInzynierska/TrainingData_NewBG'
tagged_data_dir = '/home/pawel/PracaInzynierska/LiterkiTagged_NewBG'

resources_dir = '/home/pawel/PracaInzynierska/Normalizer'

tensors_dir = resources_dir + '/Tensors'
models_dir = resources_dir + '/Models'


# If you want to use an existing model to plot graphs, predict or analize data set this to false and fill below paths
retrain_model = True

model_class = UNetModel
model_path = models_dir + '/f2051_UNetModel_e003'
history_path = models_dir + '/f2051_UNetModel_history.p'

if __name__ == "__main__":

    # 1: -- Load images and convert to tensors --

    in_res = 128
    out_res = 128

    training_data_tensor = tensors_dir + f'/training_tensor_{str(in_res)}.tsr'
    tagged_data_tensor = tensors_dir + f'/tagged_tensor_{str(out_res)}.tsr'

    if reload_images:

        seed = 69

        org_aug = DirectoryImageProvider(training_data_dir, TrainingImage)
        org_aug.shuffle(seed=seed)
        org_tsr = TensorBuilder(org_aug.all, shape=(in_res, in_res), grayscale=True).build().type('float32').range(0, 1)

        with open(training_data_tensor, 'wb') as training_file:
            dump(org_tsr, training_file)

        tgg_aug = DirectoryImageProvider(tagged_data_dir, TaggedImage)
        tgg_aug.shuffle(seed=seed)
        tgg_tsr = TensorBuilder(tgg_aug.all, shape=(out_res, out_res), grayscale=True).build().type('float32').range(0, 1)

        with open(tagged_data_tensor, 'wb') as tagged_file:
            dump(tgg_tsr, tagged_file)

    else:

        print(f'Loading tensors {training_data_tensor} and {tagged_data_tensor}...')

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

    # 3: -- Train the network or reclaim a model from storage --

    model: Model

    if retrain_model:

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
                        callbacks=[ModelCheckpoint(filepath=f'{models_dir}/{model_hash}_{model_class.__name__}_e{{epoch:03d}}',
                                                   period=save_every_n_epoch)])

        # Save the final model if it did not save before
        if not isfile(f'{models_dir}/{model_hash}_{model_class.__name__}_e{epochs_num:03d}'):
            model.save(filepath=f'{models_dir}/{model_hash}_{model_class.__name__}_e{epochs_num:03d}')

        # Save history
        with open(f'{models_dir}/{model_hash}_{model_class.__name__}_history.p', 'wb') as history_file:
            dump(history, history_file)

    else:

        model = keras.models.load_model(model_path)
        with open(history_path, 'rb'):
            history = load(history_path)
