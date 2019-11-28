import multiprocessing
import queue
from multiprocessing import Queue, Process
import keras
import numpy as np
from pickle import load
import random
import time


class ClassifierAdapter:
    correct_shape = (128, 128)

    def __init__(self, settings):
        self._master_queue: Queue = settings['master_queue']
        self.model_path = settings['classifier_settings']['model_path']
        self._prediction_queue = Queue()

        classifier_process = Process(target=self.classifier_worker)
        classifier_process.start()

    def classify(self, input_tensor: np.ndarray):
        try:
            self._prediction_queue.put(input_tensor, block=False)
        except queue.Full:
            pass

    def classifier_worker(self):
        model: keras.Model = keras.models.load_model(self.model_path)
        self._master_queue.put(('classifier_ready', None))

        while True:
            input_tensor = self._prediction_queue.get()
            if input_tensor.shape != self.correct_shape:
                raise RuntimeError("Wrong tensor shape!")

            input_tensor /= input_tensor.max()
            input_tensor = input_tensor.reshape(1, 128, 128, 1)
            prediction = model.predict(input_tensor)

            message = (
                'sign_classified',
                prediction
            )

            self._master_queue.put(message)


class ClassifierAdapterMockup(ClassifierAdapter):

    def __init__(self, settings):
        super().__init__(settings)
        self.images_tensor_path = 'C:/Users/jakub/Desktop/Inzynierka/Tensors/mockup_tensor.tsr'

        loop_process = multiprocessing.Process(target=self.module_loop, args=(self._master_queue,))
        loop_process.start()

    def module_loop(self, queue: Queue):
        with open(self.images_tensor_path, 'rb') as images_tensor_file:
            images_tensor = load(images_tensor_file)

        while True:
            time.sleep(1)
            index = random.randint(0, len(images_tensor) - 1)
            input_tensor = images_tensor[index]
            input_tensor = input_tensor.reshape(1, 128, 128, 1)
            prediction = self.model.predict(input_tensor)

            message = (
                'sign_classified',
                prediction
            )
            queue.put(message)