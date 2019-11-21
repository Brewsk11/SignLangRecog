from time import sleep
from os import listdir
from PIL.Image import Image as PILImage, open as PILOpen
from multiprocessing import Process
from random import sample


class DetectorAdapter:

    def __init__(self, on_frame_loaded_callback, on_hand_detected_callback):
        self._on_frame_loaded_callback = on_frame_loaded_callback
        self._on_hand_detected_callback = on_hand_detected_callback


class DetectorAdapterMockup(DetectorAdapter):
    mock_images_path = '/home/pawel/PycharmProjects/SignLangRecog/src/App/Modules/MockupHandImages'
    mock_images = []

    def __init__(self, on_frame_loaded_callback, on_hand_detected_callback):
        super().__init__(on_frame_loaded_callback, on_hand_detected_callback)

        # Load all images from mock images path to be displayed
        img_name_list = listdir(self.mock_images_path)

        for img_name in img_name_list:
            img_path = self.mock_images_path + '/' + img_name
            self.mock_images.append(PILOpen(img_path))

        loop_process = Process(target=self.module_loop)
        loop_process.start()

    def module_loop(self):

        while True:
            sleep(2)

            img_to_send = sample(self.mock_images, 1)

            self._on_hand_detected_callback(img_to_send)
