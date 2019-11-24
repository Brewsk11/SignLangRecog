from MainWindow import MainWindow
from multiprocessing import Queue
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from Modules.NormalizerAdapter import NormalizerAdapter
from Modules.DetectorAdapter import DetectorAdapter
from Modules.ClassifierAdapter import ClassifierAdapter
import json

settings_path = './settings.json'

if __name__ == "__main__":

    # Load settings from disk and add the master queue
    settings: dict

    with open(settings_path, 'r') as settings_file:
        settings = json.load(settings_file)

    task_queue = Queue()
    settings['master_queue'] = task_queue

    detector = DetectorAdapter(settings)
    # normalizer = NormalizerAdapter(settings)
    # classifier = ClassifierAdapter(settings)

    app = MainWindow()

    module_ready = {
        'detector': False,
        'normalizer': False,
        'classifier': False,
        'system': False
    }

    while True:
        app.update_idletasks()
        app.update()

        while not task_queue.empty():

            message, payload = task_queue.get(block=False)
            print("Oh, got a message!: " + message)

            if module_ready['system']:
                if message == "hand_detected":
                    app.on_hand_detected(payload)
                    # normalizer.normalize(payload)

                elif message == "hand_normalized":
                    app.on_hand_normalized(payload)
                    classifier.classify(payload)

                elif message == "sign_classified":
                    app.on_letter_classified(payload)

                elif message == "video_frame":
                    app.on_new_frame(payload)

            else:
                if message == "detector_ready":
                    module_ready['detector'] = True
                # elif message == "normalizer_ready":
                #     module_ready['normalizer'] = True
                # elif message == "classifier_ready":
                #     module_ready['classifier'] = True

                if module_ready['detector']:
                    module_ready['system'] = True
                    print('System ready!')
                else:
                    module_ready['system'] = False

                # if module_ready['detector'] and \
                #    module_ready['normalizer'] and \
                #    module_ready['classifier']:
                #     module_ready['system'] = True
                #     print('System ready!')
                # else:
                #     module_ready['system'] = False
