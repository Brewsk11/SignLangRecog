from multiprocessing import Queue
import keras
import numpy as np

class ClassifierAdapter:
    model_path = ''

    def __init__(self, message_queue):
        self._message_queue: Queue = message_queue
        self.model: keras.Model = keras.models.load_model(self.model_path)
        self.correct_shape = (1, 27)

    def classify(self, input_tensor: np.ndarray):
        prediction = np.zeros(1, 27)
        if (prediction.shape != self.correct_shape):
            raise RuntimeError("Wrong tensor shape!")

        prediction = self.model.predict(input_tensor)

        message = (
            'sign_classified',
            prediction
        )

        self._message_queue(message)