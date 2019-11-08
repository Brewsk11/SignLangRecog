from typing import Union, Optional
from numpy import array, ndarray
import numpy



class LettersTensorBuilder:
    """Tensor-building class operating on TaggedImage and TrainingImage objects"""

    _tensor: Optional[ndarray]
    _img_list: list
    _shape: tuple
    _grayscale: bool
    _length: int

    def __init__(self, img_list:  list):

        # Set number of samples
        self._length = len(img_list)
        self._img_list = img_list
        self._tensor = None

    def build(self) -> 'LettersTensorBuilder':
        self._tensor = self.__list_to_letter_tensor(self._img_list)

        return self

    def __list_to_letter_tensor(self, img_list: list) -> ndarray:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O',
                   'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        tensor_list = []

        print(f'{self.__class__.__name__}: Building letter tensor...')

        i = 1
        img_list_len = len(img_list)

        for img in img_list:
            letter_list = []
            for letter in letters:
                if img.letter is letter:
                    letter_list.append(1)
                else:
                    letter_list.append(0)
            # Update progress
            if i % 500 == 0 or i == img_list_len:
                print(f'{self.__class__.__name__}: [{i}/{img_list_len}]')
            i += 1
            tensor_list.append(letter_list)

        tensor_arr = array(tensor_list, dtype=numpy.uint8)

        return tensor_arr

    def size(self, val: tuple) -> 'LettersTensorBuilder':
        """Set whether to convert to grayscale when building tensor"""
        if self._tensor is not None:
            raise RuntimeError('Setting grayscale should happen before building tensor')
        self._shape = val
        return self

    def img_list(self, val: list) -> 'LettersTensorBuilder':
        """Set a new source of images. The tensor will have to be rebuilt after setting this property"""
        if self._tensor is not None:
            raise RuntimeWarning('A tensor has already been built. You\'ll have to build it again.')
        self._img_list = val
        return self

    @property
    def tensor(self) -> ndarray:
        """Return previously built tensor"""
        if self._tensor is None:
            raise RuntimeError('You have to build the tensor before you can access it')
        return self._tensor

    @property
    def length(self) -> int:
        """Numbers of samples in the provided image list"""
        return self._length
