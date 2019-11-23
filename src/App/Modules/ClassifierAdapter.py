from multiprocessing import Queue
import keras
import numpy as np

class ClassifierAdapter:

    def __init__(self, settings):
        self._master_queue: Queue = settings['master_queue']
        self.model_path = settings['classifier_settings']['model_path']
        self.model: keras.Model = keras.models.load_model(self.model_path)
        self.correct_shape = (128, 128)

        self._master_queue.put(('classifier_ready', None))

    def classify(self, input_tensor: np.ndarray):
        if input_tensor.shape != self.correct_shape:
            raise RuntimeError("Wrong tensor shape!")

        input_tensor = input_tensor.reshape(1, 128, 128, 1)
        prediction = self.model.predict(input_tensor)

        message = (
            'sign_classified',
            prediction
        )

        self._master_queue.put(message)