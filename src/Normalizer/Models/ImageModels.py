from PIL.Image import Image as PILImage, open as PILOpen, BICUBIC, BILINEAR, LANCZOS, NEAREST
from os.path import abspath
import re as regex


class ImageAbstr:
    """
    Abstract base class for TrainingImage and TaggedImage classes
    """

    def __init__(self, filepath: str, filename_pattern: str):
        """
        Pulls the image from filesystem and stores it in pillow object; parses image filename and fills data properties

        :param filepath: Path to the image, may be relative
        :raises AttributeError: When some image in the directory is not compliant with the filename pattern
        """
        self._width: int
        self._height: int
        self._pillow_image: PILImage
        self._full_path: str
        self._path: str
        self._filename: str
        self._letter: str
        self._extension: str
        self._number: int
        self._filename_pattern: str

        # Change relative path to absolute path
        filepath = abspath(filepath)

        # Initialize pillow image object and connected properties
        self._pillow_image = PILOpen(filepath)
        self._width, self._height = self._pillow_image.size

        # Set fullpath as path + filename
        self._full_path = filepath

        # Set path and filename separately
        var = filepath.rsplit('/', 1)
        self._path = str(var[0])
        self._filename = str(var[1])
        var = self._filename

        # Find out if the filename if compliant with tagged or training image naming conventions
        if regex.match(filename_pattern, self._filename) is None:
            raise AttributeError('Provided filename ' + self._filename + ' is not compliant with naming pattern'
                                 + self._filename_pattern)

        # Set letter
        self._letter = var[0]
        var = var[1:]

        # Set extension
        var = var.rsplit('.', 2)
        if len(var) == 2:
            self._extension = var[1]
            var = var[0]

        # Extract image number from the filename
        if var.find('_') > 0:
            var = var.split('_')
            var = var[0]

        # Set image number
        self._number = int(var)

    def resize(self, size: tuple, inplace: bool = False) -> PILImage:
        """Resizes and returns pillow object

        :param size: A tuple (width, height)
        :param inplace: 'True' means that the resized image will replace current pillow image object
        """
        new_pillow_image = self._pillow_image.resize(size, resample=BICUBIC)
        if inplace:
            self._pillow_image = new_pillow_image
        return new_pillow_image

    @property
    def width(self) -> int:
        """Width of the image taken from the loaded pillow image object"""
        return self._pillow_image.size[0]

    @property
    def height(self) -> int:
        """Height of the image taken from the loaded pillow image object"""
        return self._pillow_image.size[1]

    @property
    def pillow_image(self) -> PILImage:
        """Pillow image object loaded from the provided filepath in the constructor"""
        return self._pillow_image

    @property
    def full_path(self) -> str:
        """Absolute path to the image including the image's filename"""
        return self._full_path

    @property
    def path(self) -> str:
        """Absolute path to the image not including the image's filename, without the trainling '/'"""
        return self._path

    @property
    def filename(self) -> str:
        """Image's filename incuding the extension"""
        return self._filename

    @property
    def letter(self) -> str:
        """The letter the image was described by. Should be uppercase A-Z for letters, 'n' for nothing, 's' for space"""
        return self._letter

    @property
    def extension(self) -> str:
        """Extension of the image's file"""
        return self._extension

    @property
    def number(self) -> int:
        """The number that was included in the filename"""
        return self._number


class TaggedImage(ImageAbstr):
    """
    The TaggedImage class is a model for the images in the tagged set.To successfully load the image the filename
    must consist of a letter [A-Zns] at the start, a number, '_' char,
    image size and extension (ex. H1234_128.jpg, s4231_16.bmp).
    """

    def __init__(self, filepath: str):

        self._resolution: int

        self._filename_pattern = "^[A-Zdns][0-9]{1,6}_[0-9]{1,3}\.(bmp|jpg)$"
        super().__init__(filepath, self._filename_pattern)

        self._resolution = int(self.filename.rsplit('_', 1)[1].split('.')[0])
        if self._resolution != self.width and self._resolution !=  self.height:
            raise RuntimeError(f'Resolution specified in filename {self.filename}'
                               f'is not the same as image\'s width and height: ({self.width} {self.height}))')

    @property
    def resolution(self) -> int:
        """Returns resolution specified in the filename"""
        return self._resolution


class TrainingImage(ImageAbstr):
    """
    The TrainingImage class is a model for the images in the training set. To successfully load the image the filename
    must consist of a letter [A-Zns] at the start, a number and extension (ex. H1234.jpg, s4231.bmp).
    """

    def __init__(self, filepath: str):
        self._filename_pattern = "^[A-Zdns][0-9]{1,6}\.(bmp|jpg)$"
        super().__init__(filepath, self._filename_pattern)

