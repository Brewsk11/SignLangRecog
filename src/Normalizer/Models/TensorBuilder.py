from typing import Union, Optional
from numpy import array, ndarray
import numpy

from Normalizer.Models.ImageModels import TaggedImage, TrainingImage


class TensorBuilder:
    """Tensor-building class operating on TaggedImage and TrainingImage objects"""

    _tensor: Optional[ndarray]
    _img_list: list
    _shape: tuple
    _grayscale: bool
    _length: int

    def __init__(self, img_list:  list, shape=None, grayscale=False):

        # Check if every image is the same size
        width, height = img_list[0].width, img_list[0].height
        for img in img_list:
            if img.width != width or img.height != height:
                raise ValueError(f'Size of image {img.filename} is different from others\''
                                 f'(expected ({width}, {height}), got {img.width}, {img.height}))')

        # If no size specified default to first image's size
        if shape is None:
            shape = (img_list[0].width, img_list[0].height)

        # Set number of samples
        self._length = len(img_list)

        self._img_list = img_list
        self._shape = shape
        self._grayscale = grayscale
        self._tensor = None

    def build(self) -> 'TensorBuilder':
        """
        Converts an image object to a tensor. Takes properties from previously set values.

        :raises AttributeError: When obj is not a TaggedImage, TrainingImage or a list
        :return: Itself
        """

        if type(self._img_list) == TaggedImage or type(self._img_list) == TrainingImage:
            self._tensor = self.__img_to_tensor(self._img_list, self._shape, self._grayscale, numpy_out=True)
        elif type(self._img_list) == list:
            self._tensor = self.__list_to_tensor(self._img_list, self._shape, self._grayscale)
        else:
            raise AttributeError("Parameter obj is neither a list nor a pillow image object")

        return self

    def __list_to_tensor(self, img_list: list, tensor_size: tuple, grayscale: bool) -> ndarray:
        """
        Function converting a list of TrainingImage or TaggedImage to a tensor as a numpy.ndarray object

        :param img_list: A list of TaggedImage or TrainingImage objects
        :param tensor_size: A (width, height) tuple
        :param grayscale: If true the resulting list's images will be converted to 1-channel grayscale tensor
        (width, height, 1) instead of 3-channel (width, height, 3)
        :return: A tensor of type numpy.ndarray, of size (len(img_list), width, height)
        """
        tensor_list = []

        print(f'{self.__class__.__name__}: Building tensor...')

        i = 1
        img_list_len = len(img_list)

        for img in img_list:
            img_as_tensor = self.__img_to_tensor(img, tensor_size, grayscale, numpy_out=False)
            tensor_list.append(img_as_tensor)

            # Update progress
            if i % 500 == 0 or i == img_list_len:
                print(f'{self.__class__.__name__}: [{i}/{img_list_len}]')
            i += 1

        tensor_arr = array(tensor_list, dtype=numpy.uint8)

        # Reshape the tensor to (samples, width, height, channels) explicitly.
        # When grayscale == True it's (samples, width, height)
        if grayscale:
            tensor_arr = tensor_arr.reshape((len(img_list), tensor_size[0], tensor_size[1], 1))

        return tensor_arr

    def __img_to_tensor(self, img: Union[TrainingImage, TaggedImage], tensor_size: tuple, grayscale: bool,
                        numpy_out: bool) -> Union[ndarray, list]:
        """
        Function turning a TaggedImage/TrainingImage to a list or numpy.ndarray tensor

        :param img: TaggedImage or TrainingImage object
        :param tensor_size: (width, height) tuple
        :param grayscale: If true the result will be converted to 1-channel grayscale tensor (width, height, 1) instead
        of 3-channel (width, height, 3)
        :param numpy_out: If true the returned object will be of type numpy.ndarray instead of list; grayscale output
        will have the pixel values as a number, non-grayscale output will have pixels as a 3-size tuple (r, g, b)
        :return: A 2-dimensional list ar numpy.ndarray of pixel values
        """

        # Resize the image to match the tensor_size for straight-forward conversion
        pil_img = img.resize(tensor_size)

        # Convert to grayscale
        if grayscale:
            pil_img = pil_img.convert('L')

        width, height = tensor_size
        img_data = pil_img.getdata()

        matrix = []

        for row in range(height):
            pixel_row = []

            for column in range(width):

                # If grayscale: 'pixel' is a number, else: 'pixel' is a (r, g, b) tuple
                pixel = img_data[column + row * width]
                pixel_row.append(pixel)

            matrix.append(pixel_row)

        if numpy_out:
            # Numpy ndarray converts the non-grayscale pixel (r, g, b) tuples to [r, g, b] lists
            out = array(matrix)
        else:
            out = matrix

        return out

    def type(self, new_type: str) -> 'TensorBuilder':
        """
        Changes tensor type as in ndarray specification

        :param new_type: New type of values in the tensor as a string
        :return: Itself
        """

        self._tensor = self._tensor.astype(new_type)
        return self

    def range(self, new_min: float, new_max: float) -> 'TensorBuilder':
        """
        Shifts tensor values to match range [min_val, max_val]

        :param new_max: Minimum value
        :param new_min: Maximum value
        :return: An ndarray with changed values range
        """
        if self._tensor is None:
            raise RuntimeError('You have to build tensor before you can change the range')

        min_ = self._tensor.flatten().min()
        max_ = self._tensor.flatten().max()

        range_ = max_ - min_
        new_range = new_max - new_min

        out = self._tensor

        out -= min_
        out *= new_range / range_
        out += new_min

        self._tensor = out
        return self

    def grayscale(self, val: bool) -> 'TensorBuilder':
        """Set if to convert to grayscale when building tensor"""
        if self._tensor is not None:
            raise RuntimeError('Setting grayscale should happen before building tensor')
        self._grayscale = val
        return self

    def size(self, val: tuple) -> 'TensorBuilder':
        """Set whether to convert to grayscale when building tensor"""
        if self._tensor is not None:
            raise RuntimeError('Setting size should happen before building tensor')
        self._shape = val
        return self

    def img_list(self, val: list) -> 'TensorBuilder':
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

