from time import sleep
from os import listdir
from PIL.Image import Image as PILImage, open as PILOpen
from multiprocessing import Process, Queue
import random
import matplotlib.pyplot as plt


class DetectorAdapter:

    def __init__(self, settings):
        self.settings: dict = settings


class DetectorAdapterMockup(DetectorAdapter):
    mock_images = []

    def __init__(self, settings):
        super().__init__(settings)
        self.mock_images_path = self.settings['mock_images_path']

        self._message_queue = settings['master_queue']

        # Load all images from mock images path to be displayed
        img_name_list = listdir(self.mock_images_path)

        for img_name in img_name_list:
            img_path = self.mock_images_path + '/' + img_name

            img = PILOpen(img_path)
            self.mock_images.append(img)

        loop_process = Process(target=self.module_loop, args=(self._message_queue,))
        loop_process.start()

    def module_loop(self, queue: Queue):

        while True:
            sleep(2)
            index = random.randint(0, len(self.mock_images) - 1)

            message = (
                "hand_detected",
                self.mock_images[index]
            )

            queue.put(message)
