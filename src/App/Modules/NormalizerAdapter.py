from time import sleep
from os import listdir
from PIL.Image import Image as PILImage, open as PILOpen
from multiprocessing import Process, Queue
import random
import matplotlib.pyplot as plt
import keras
from Normalizer.Models.TensorBuilder import TensorBuilder

class NormalizerAdapter:

    model_path = '/home/pawel/PracaInzynierska/Normalizer/Models/dad07_UNetModel_e002'

    def __init__(self, message_queue):
        self._message_queue: Queue = message_queue

        self.model = keras.models.load_model(self.model_path)
        # self.tensor_builder =

    def normalize(self, hand_photo):
        pass