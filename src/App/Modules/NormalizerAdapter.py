from time import sleep
from os import listdir
from PIL.Image import Image as PILImage, open as PILOpen
from multiprocessing import Process, Queue
import random
import matplotlib.pyplot as plt
import keras
import numpy as np
from Normalizer.Models.TensorBuilder import TensorBuilder
import time


class NormalizerAdapter:
    model_path = '/home/pawel/PracaInzynierska/Normalizer/Models/dad07_UNetModel_e002'
    tensor_size = (128, 128)

    def __init__(self, message_queue):
        self._message_queue: Queue = message_queue

        self.model: keras.Model = keras.models.load_model(self.model_path)

    def normalize(self, hand_photo: PILImage):

        tsr_img = self.image_to_tensor(hand_photo)
        tsr_arr = tsr_img.reshape((1, self.tensor_size[0], self.tensor_size[1], 1))

        t0 = time.clock()
        pred = self.model.predict(tsr_arr)
        t1 = time.clock()
        elapsed = t1 - t0
        print(f"Prediction complete in {elapsed}")  # TODO: Change to time elapsed?

        pred = pred.reshape((128, 128))

        message = (
            'hand_normalized',
            pred
        )

        self._message_queue.put(message)

    def normalizer_worker(self, hand_photo: PILImage):
        pass

    def image_to_tensor(self, pil_img):

        tensor_size = self.tensor_size

        # Resize the image to match the tensor_size for straight-forward conversion
        pil_img = pil_img.resize(tensor_size)

        # Convert to grayscale
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

        # Numpy ndarray converts the non-grayscale pixel (r, g, b) tuples to [r, g, b] lists
        out = np.array(matrix)

        return out
