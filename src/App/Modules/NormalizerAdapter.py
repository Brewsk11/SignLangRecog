import time
import queue
from multiprocessing import Queue, Process

import keras
import numpy as np
from PIL.Image import Image as PILImage

import time

import matplotlib.pyplot as plt

class NormalizerAdapter:
    tensor_size = (128, 128)

    def __init__(self, settings):
        self._master_queue: Queue = settings['master_queue']
        self.model_path = settings['normalizer_settings']['model_path']

        self._prediction_queue = Queue(maxsize=1)

        normalizer_process = Process(target=self.normalizer_worker)
        normalizer_process.start()

    def normalize(self, hand_photo: PILImage):
        try:
            self._prediction_queue.put(hand_photo, block=False)
        except queue.Full:
            pass

    def normalizer_worker(self):

        model: keras.Model = keras.models.load_model(self.model_path)

        end = time.time()
        self._master_queue.put(('normalizer_ready', end))

        while True:
            hand_photo_time = self._prediction_queue.get(block=True)
            hand_photo_time.normalizer_start_time = time.time()
            hand_photo = hand_photo_time.img

            tsr_img = self.image_to_tensor(hand_photo)
            tsr_arr: np.ndarray = tsr_img.reshape((1, self.tensor_size[0], self.tensor_size[1], 1))

            tsr_arr = tsr_arr.astype('float32')
            tsr_arr -= tsr_arr.min()
            tsr_arr /= tsr_arr.max()

            pred = model.predict(tsr_arr)
            
            pred = pred.reshape((128, 128))
            pred /= pred.max()

            hand_photo_time.img = pred
            hand_photo_time.normalizer_end_time = time.time()
            message = (
                'hand_normalized',
                hand_photo_time
            )

            self._master_queue.put(message)

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
