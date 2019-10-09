import keras
import matplotlib.pyplot as plt
from pickle import load as pickle_load

from Normalizer.Providers.DirectoryImageProvider import DirectoryImageProvider
from Normalizer.Models.TensorBuilder import TensorBuilder
from Normalizer.Models.ImageModels import TrainingImage, TaggedImage

resources_dir = '/home/pawel/PracaInzynierska/Normalizer'

tensors_dir = resources_dir + '/Tensors'
models_dir = resources_dir + '/Models'
test_imgs_path = resources_dir + '/TestData'

model_path = models_dir + '/dad07_UNetModel_e00'
history_path = models_dir + '/f2051_UNetModel_history.p'


def show_on_plot(pred, org):
    num = len(pred)
    plt.figure(figsize=(20, 10))
    for j in range(num):
        plt.subplot(2, num, 1 + j)
        plt.imshow(pred[j].reshape((128, 128)))
        plt.subplot(2, num, num + 1 + j)
        plt.imshow(org[j].reshape((128, 128)))
    plt.show()


if __name__ == "__main__":

    predictions = []

    ip = DirectoryImageProvider(test_imgs_path, TrainingImage)
    ip.shuffle(seed=1, inplace=True)
    org_tsr = TensorBuilder(ip.all, shape=(128, 128), grayscale=True).build().type('float32').range(0, 1)

    for i in range(3):
        model = keras.models.load_model(model_path + str(i + 1))
        with open(history_path, 'rb') as history_file:
            history = pickle_load(history_file)

        predictions.append(model.predict(org_tsr.tensor))

    for i in range(1, 5):
        show_on_plot(predictions[0][i*10:(i+1)*10], org_tsr.tensor[i*10:(i+1)*10])
        show_on_plot(predictions[1][i*10:(i+1)*10], org_tsr.tensor[i*10:(i+1)*10])
        show_on_plot(predictions[2][i*10:(i+1)*10], org_tsr.tensor[i*10:(i+1)*10])
