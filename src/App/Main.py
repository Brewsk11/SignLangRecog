from MainWindow import MainWindow
from multiprocessing import Queue
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from Modules.NormalizerAdapter import NormalizerAdapter
from Modules.DetectorAdapter import DetectorAdapter
from Modules.ClassifierAdapter import ClassifierAdapter
import json

import time

settings_path = './settings.json'

if __name__ == "__main__":

    # Load settings from disk and add the master queue
    settings: dict

    with open(settings_path, 'r') as settings_file:
        settings = json.load(settings_file)

    task_queue = Queue()
    system_ready = Queue()
    settings['master_queue'] = task_queue
    settings['system_ready'] = system_ready
    detector = DetectorAdapter(settings)
    normalizer = NormalizerAdapter(settings)
    classifier = ClassifierAdapter(settings)

    app = MainWindow()

    module_ready = {
        'detector': False,
        'normalizer': False,
        'classifier': False,
        'system': False
    }

    f = open("..//Tests//hand_classify.txt", "a+")
    f.write("letter classify\n")
    tmp = 0
    z_count = 0
    while True:
        app.update_idletasks()
        app.update()

        while not task_queue.empty():

            message, payload = task_queue.get(block=False)
            print("Oh, got a message "+str(tmp)+"!:" + message)


            if module_ready['system']:
                if message == "hand_detected":
                    app.on_hand_detected(payload.img)
                    normalizer.normalize(payload)

                if message == "hand_normalized":
                    app.on_hand_normalized(payload.img)
                    classifier.classify(payload)

                elif message == "sign_classified":
                    tmp += 1
                    payload_bak = payload
                    pred_list = payload.img
                    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                              'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                    pred_list = pred_list.flatten()
                    classified_letter = labels[pred_list.argmax()]

                    result = 0
                    if payload.letter == classified_letter:
                        result = 1
                    f.write(payload.letter + " " + str(result) + "\n")
                    if payload.letter == "Z":
                        z_count += 1
                        if z_count == 100:
                            f.close()
                            print("Koniec, dziekuje")
                    app.on_letter_classified(payload_bak.img)

                # if message == "video_frame":
                #     f.write(payload.letter+" "+str(payload.detect)+"\n")
                #     app.on_new_frame(payload.img)

            else:
                if message == "detector_ready":
                    module_ready['detector'] = True
                elif message == "normalizer_ready":
                    module_ready['normalizer'] = True
                elif message == "classifier_ready":
                    module_ready['classifier'] = True

                if module_ready['detector'] and \
                   module_ready['normalizer'] and \
                   module_ready['classifier']:
                    module_ready['system'] = True
                    print('System ready!')
                    settings['system_ready'].put("ready")
                else:
                    module_ready['system'] = False
