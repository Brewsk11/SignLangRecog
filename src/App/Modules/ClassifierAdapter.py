from multiprocessing import Queue
import keras
import numpy as np

class ClassifierAdapter:

    def __init__(self, settings):
        self._master_queue: Queue = settings['master_queue']
        self.model_path = ['classfiier_settings']['model_path']
        self.model: keras.Model = keras.models.load_model(self.model_path)
        self.correct_shape = (1, 27)

    def classify(self, input_tensor: np.ndarray):
        if (input_tensor.shape != self.correct_shape):
            raise RuntimeError("Wrong tensor shape!")

        prediction = self.model.predict(input_tensor)

        message = (
            'sign_classified',
            prediction
        )

        self._message_queue.put(message)