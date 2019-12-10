from pickle import dump, load
import numpy
from random import shuffle
from Common.DirectoryImageProvider import DirectoryImageProvider
from Common.Models.TensorBuilder import TensorBuilder
from Common.Models.ImageModels import TaggedImage
from Classifier.Models.LettersTensorBuilder import LettersTensorBuilder

generate_new_or_connect = True
res = 128
seed = 69

tensors_dir = 'C:/Users/jakub/Desktop/Inzynierka/Tensors/Refactor/'
tagged_data_dir = 'LiterkiTest'
tagged_data_dir2 = 'LiterkiTrain2'
tagged_data_path = 'C:/Users/jakub/Desktop/Inzynierka/' + tagged_data_dir + '/'
tagged_data_path2 = 'C:/Users/jakub/Desktop/Inzynierka/' + tagged_data_dir2 + '/'

# if flag 'generate_new_or_connect' is set to False, the script connects two pairs of tensors instead of generating a new pair
# this was needed due to Windows throwing error in case of generating tensors from too many images at the same time
if __name__ == "__main__":
    if generate_new_or_connect:
        #generate first pair of tensors
        images_tensor_path = tensors_dir + tagged_data_dir + '_images.tsr'
        letters_tensor_path = tensors_dir + tagged_data_dir + '_letters.tsr'

        tagged_images = DirectoryImageProvider(tagged_data_path, TaggedImage, res)
        tagged_images.shuffle(seed=seed)

        images_tensor = TensorBuilder(tagged_images.resolution(res), shape=(res, res), grayscale=True).build().type('float32').range(0, 1)
        letters_tensor = LettersTensorBuilder(tagged_images.resolution(res)).build()

        with open(images_tensor_path, 'wb') as images_file:
            dump(images_tensor, images_file)

        with open(letters_tensor_path, 'wb') as letters_file:
            dump(letters_tensor, letters_file)

        #generate second pair of tensors
        images_tensor_path = tensors_dir + tagged_data_dir2 + '_images.tsr'
        letters_tensor_path = tensors_dir + tagged_data_dir2 + '_letters.tsr'

        tagged_images = DirectoryImageProvider(tagged_data_path2, TaggedImage, res)
        tagged_images.shuffle(seed=seed)
        images_tensor = TensorBuilder(tagged_images.resolution(res), shape=(res, res), grayscale=True).build().type('float32').range(0, 1)
        letters_tensor = LettersTensorBuilder(tagged_images.resolution(res)).build()

        with open(images_tensor_path, 'wb') as images_file:
            dump(images_tensor, images_file)

        with open(letters_tensor_path, 'wb') as letters_file:
            dump(letters_tensor, letters_file)

    else:
        print("Loading part tensors")
        images_tensor_path = tensors_dir + tagged_data_dir + '_images.tsr'
        images_tensor_path2 = tensors_dir + tagged_data_dir2 + '_images.tsr'
        letters_tensor_path = tensors_dir + tagged_data_dir + '_letters.tsr'
        letters_tensor_path2 = tensors_dir + tagged_data_dir2 + '_letters.tsr'

        with open(images_tensor_path, 'rb') as images_file:
            images_tensor1 = load(images_file)

        with open(images_tensor_path2, 'rb') as images_file:
            images_tensor2 = load(images_file)

        with open(letters_tensor_path, 'rb') as letters_file:
            letters_tensor1 = load(letters_file)

        with open(letters_tensor_path2, 'rb') as letters_file:
            letters_tensor2 = load(letters_file)

        print("Connecting tensors")

        images_tensor1._tensor = numpy.concatenate((images_tensor1.tensor, images_tensor2.tensor))
        images_tensor1._length = images_tensor1._length + images_tensor2._length
        images_tensor1._img_list.extend(images_tensor2._img_list)
        letters_tensor1._tensor = numpy.concatenate((letters_tensor1.tensor, letters_tensor2.tensor))
        letters_tensor1._length = letters_tensor1._length + letters_tensor2._length
        letters_tensor1._img_list.extend(letters_tensor2._img_list)

        #shuffle connected tensors
        print("Shuffle connected tensors")
        shuffled_list1 = []
        shuffled_list2 = []
        index_shuf = list(range(images_tensor1._length))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_list1.append(images_tensor1._img_list[i])
            shuffled_list2.append(letters_tensor1._img_list[i])

        images_tensor1._img_list = shuffled_list1
        letters_tensor1._img_list = shuffled_list2

        length = letters_tensor1._length

        shuffled_list1 = numpy.zeros(shape=(length, 128, 128, 1))
        shuffled_list2 = numpy.zeros(shape=(length, 27))
        j = 0
        for i in index_shuf:
            shuffled_list1[j] = images_tensor1._tensor[i]
            shuffled_list2[j] = letters_tensor1._tensor[i]
            j = j + 1

        images_tensor1._tensor = shuffled_list1
        letters_tensor1._tensor = shuffled_list2

        print("Saving full tensors")
        with open(tensors_dir + 'all_images.tsr', 'wb') as images_file:
            dump(images_tensor1, images_file)

        with open(tensors_dir + 'all_letters.tsr', 'wb') as letters_file:
            dump(letters_tensor1, letters_file)

