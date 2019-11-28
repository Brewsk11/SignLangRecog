from pickle import dump

from Common import DirectoryImageProvider
from Common.Models.TensorBuilder import TensorBuilder
from Common.Models.ImageModels import TrainingImage, TaggedImage


training_data_dir = '/home/pawel/PracaInzynierska/TrainingData_NewBG'
tagged_data_dir = '/home/pawel/PracaInzynierska/LiterkiTagged_NewBG'

resources_dir = '/home/pawel/PracaInzynierska/Normalizer'

tensors_dir = resources_dir + '/Tensors'


if __name__ == "__main__":

    in_res = 128
    out_res = 128

    training_data_tensor = tensors_dir + f'/training_tensor_{str(in_res)}.tsr'
    tagged_data_tensor = tensors_dir + f'/tagged_tensor_{str(out_res)}.tsr'

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
