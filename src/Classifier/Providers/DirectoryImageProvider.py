import random
from os import listdir
from os.path import abspath, isdir
from typing import Union, Optional

from Classifier.Models.ImageModels import TaggedImage, TrainingImage


class DirectoryImageProvider:
    """
    Class providing lists of TaggedImage or TrainingImage objects from a directory.
    """

    _path: str
    _img_type: Union[TaggedImage, TrainingImage]
    _img_list: Optional[list]
    _resolution: int

    def __init__(self, path: str, img_type: Union[TaggedImage, TrainingImage], resolution: int, recursive: bool = True):
        self._path = path
        self._img_type = img_type
        self._resolution = resolution

        # Recursively add images from the directory
        self._img_list = self.__load_img_list(self.path, recursive)

    def shuffle(self, seed: int = None, inplace: bool = True):

        # Seed the randomizer
        random.seed(seed)

        new_list = random.sample(self._img_list, len(self._img_list))

        if inplace:
            self._img_list = new_list

        return new_list

    def __load_img_list(self, path: str, recursive: bool = True) -> list:

        print(f'{self.__class__.__name__}: Loading files from {path}...')

        img_list = []
        filename_list = listdir(path)

        # Traverse each file
        for filename in filename_list:
            abs_path = path + '/' + filename

            if isdir(abs_path) and recursive:
                lower_dir_imgs = self.__load_img_list(abs_path)
                img_list += lower_dir_imgs
                continue

            img = self._img_type(abs_path, self._resolution)
            img_list.append(img)

        img_list.sort(key=lambda im: (im.letter, im.number))

        return img_list

    @property
    def path(self) -> str:
        """Returns absolute path the provider was set to get the images from"""
        return self._path

    @path.setter
    def path(self, new_path: str):
        """Sets the path of the provider; the path will be turned into an absolute path"""
        if self._path != new_path:
            self._img_list = None

        self._path = abspath(new_path)

    @property
    def img_type(self) -> Union[TaggedImage, TrainingImage]:
        """Returns the type the provider is getting the images as; can be either TaggedImages or TrainingImages"""
        return self._img_type

    @property
    def all(self) -> list:
        """Returns a list of all images loaded from the path sorted by number"""
        return self._img_list

    def resolution(self, val: int, inplace: bool = True) -> list:
        if self._img_type != TaggedImage:
            raise RuntimeError('To get images of specific resulution provider has to be '
                               'set to objects of type TaggedImage')

        list_of_res = [img for img in self._img_list if img.resolution == val]
        if len(list_of_res) == 0:
            raise RuntimeWarning(f'The list containing images from {self.path} and resolution {str(val)} is empty')

        if inplace:
            self._img_list = list_of_res

        return list_of_res
