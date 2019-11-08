from pickle import dump, load
import numpy

from Classifier.Providers.DirectoryImageProvider import DirectoryImageProvider
from Classifier.Models.TensorBuilder import TensorBuilder
from Classifier.Models.ImageModels import TaggedImage
from Classifier.Models.LettersTensorBuilder import LettersTensorBuilder

#if true generate new part tensors, if false connect tensors
generate_new_or_connect = False
res = 128
seed = 69
tagged_data_dir = 'LiterkiPawel2'

main_path = 'C:/Users/jakub/Desktop/Inżynierka/'
tensors_dir = 'Tensors/'
tagged_data_path = main_path + tagged_data_dir
images_tensor_name = f'/{tagged_data_dir}_tagged_images_{str(res)}.tsr'
letters_tensor_name = f'/{tagged_data_dir}_letters_{str(res)}.tsr'

if __name__ == "__main__":
    if(generate_new_or_connect):
        images_tensor_path = main_path + tensors_dir + images_tensor_name
        letters_tensor_path = main_path + tensors_dir + letters_tensor_name

        tagged_images = DirectoryImageProvider(tagged_data_path, TaggedImage, res)
        tagged_images.shuffle(seed=seed)
        images_tensor = TensorBuilder(tagged_images.resolution(res), shape=(res, res), grayscale=True).build().type('float32').range(0, 1)
        letters_tensor = LettersTensorBuilder(tagged_images.resolution(res)).build()

        with open(images_tensor_path, 'wb') as images_file:
            dump(images_tensor, images_file)

        with open(letters_tensor_path, 'wb') as letters_file:
            dump(letters_tensor, letters_file)

    else:
        print("Loading part tensors")
        with open(main_path + tensors_dir + 'LiterkiPawel1_tagged_images_128.tsr', 'rb') as images_file:
            images_tensor1 = load(images_file)

        with open(main_path + tensors_dir + 'LiterkiPawel2_tagged_images_128.tsr', 'rb') as images_file:
            images_tensor2 = load(images_file)

        with open(main_path + tensors_dir + 'LiterkiPawel1_letters_128.tsr', 'rb') as letters_file:
            letters_tensor1 = load(letters_file)

        with open(main_path + tensors_dir + 'LiterkiPawel2_letters_128.tsr', 'rb') as letters_file:
            letters_tensor2 = load(letters_file)

        print("Connecting tensors")
        images_tensor1._tensor = numpy.concatenate((images_tensor1.tensor, images_tensor2.tensor))
        images_tensor1._length = images_tensor1._length + images_tensor2._length
        letters_tensor1._tensor = numpy.concatenate((letters_tensor1.tensor, letters_tensor2.tensor))
        letters_tensor1._length = letters_tensor1._length + letters_tensor2._length

        images_tensor_path = main_path + tensors_dir + 'images.tsr'
        letters_tensor_path = main_path + tensors_dir + 'letters.tsr'

        print("Saving full tensors")
        with open(images_tensor_path, 'wb') as images_file:
            dump(images_tensor1, images_file)

        with open(letters_tensor_path, 'wb') as letters_file:
            dump(letters_tensor1, letters_file)

