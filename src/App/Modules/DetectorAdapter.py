from Modules.Detector.utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
import queue
import time
from Modules.Detector.utils.detector_utils import WebcamVideoStream
import datetime
import argparse
from threading import Thread
import numpy as np
from PIL import ImageTk, Image

def co_worker(input_q, output_q, box_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    # add .compat.v1. fragment to work with newer tensorflow
    sess = tf.compat.v1.Session(graph=detection_graph)
    blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    while True:
        box_image = None
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            box_image = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1

        else:
            output_q.put(frame)

        if box_image is not None and box_image.size != 0:
            box_q.put(box_image)
        # else:
        #     box_q.put(blank_image)
    sess.close()

class DetectorAdapter:

    def __init__(self, settings, score_thresh=0.5, width=300, height=200, num_workers=2, fps=1, queue_size=5, video_source=0, num_hands=1,):
        self.settings: dict = settings
        self._message_queue = settings['master_queue']
        args = {"width": width, "height": height, "num_workers": num_workers, "fps": fps, "queue_size": queue_size, "video_source": video_source, "num_hands": num_hands}
        self.input_q = multiprocessing.Queue(maxsize=args["queue_size"])
        self.output_q = multiprocessing.Queue(maxsize=args["queue_size"])
        self.box_q = multiprocessing.Queue(maxsize=args["queue_size"])

        self.video_capture = WebcamVideoStream(
            src=args["video_source"], width=args["width"], height=args["height"]).start()

        cap_params = {}
        frame_processed = 0
        cap_params['im_width'], cap_params['im_height'] = self.video_capture.size()
        cap_params['score_thresh'] = score_thresh

        # max number of hands we want to detect/track
        cap_params['num_hands_detect'] = args["num_hands"]

        print(cap_params, args)

        # spin up workers to paralleize detection.
        self.pool = multiprocessing.Pool(args["num_workers"], co_worker,
                    (self.input_q, self.output_q, self.box_q, cap_params, frame_processed))

        self.start_time = datetime.datetime.now()
        self.num_frames = 0
        self.fps = 0
        self.index = 0
        self.frame = None
        self.box_image = None
        self.blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        self.stopped = False
        self.except_table = []
        Thread(target=self.update, args=(self._message_queue,)).start()

    def update(self, queue: multiprocessing.Queue):
        while True:
            if self.stopped:
                print("Detector stop")
                return
            try:
                frame = self.video_capture.read()
                frame = cv2.flip(frame, 1)
                self.index += 1
            except:
                self.except_table.append("video read exception")
                frame = self.blank_image

            try:
                self.input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                output_frame = self.output_q.get()

                pil_frame = Image.fromarray(output_frame)
                message = (
                    "video_frame",
                    pil_frame
                )
                queue.put(message)
            except:
                self.except_table.append("Queue excpetion")
                output_frame = self.blank_image
            
            try:
                self.box_image = self.box_q.get(False)
                pil_hand = Image.fromarray(self.box_image)

                message = (
                    "hand_detected",
                    pil_hand
                )
                queue.put(message)
            except:
                self.except_table.append("No image in box queue")
                

            try:
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            except:
                self.except_table.append("cvtColor exception")

            try:
                self.elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
                self.num_frames += 1
                self.fps = self.num_frames / self.elapsed_time
                # print("frame ",  index, num_frames, elapsed_time, fps)
                self.frame = output_frame
            except:
                self.except_table.append("calculations excpetion")

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.pool.terminate()
        self.video_capture.stop()
        self.elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.fps = self.num_frames / self.elapsed_time
        print("fps", self.fps)
        cv2.destroyAllWindows()

    def getFrame(self):
        # return the frame most recently read
        return self.frame

    # def size(self):
    #     # return size of the capture device
    #     return self.stream.get(3), self.stream.get(4)

    def getFps(self):
        return self.fps

    def getNumFrames(self):
        return self.num_frames

    def getStartTime(self):
        return self.start_time

    def getBoxImage(self):
        return self.box_image

    def getExceptTable(self):
        return self.except_table

    def clearExceptTable(self):
        for item in self.except_table:
            self.except_table.remove(item)


# class DetectorAdapterMockup(DetectorAdapter):
#     mock_images = []

#     def __init__(self, settings):
#         super().__init__(settings)
#         self.mock_images_path = self.settings['mock_images_path']

#         self._message_queue = settings['master_queue']

#         # Load all images from mock images path to be displayed
#         img_name_list = listdir(self.mock_images_path)

#         for img_name in img_name_list:
#             img_path = self.mock_images_path + '/' + img_name

#             img = PILOpen(img_path)
#             self.mock_images.append(img)

#         loop_process = Process(target=self.module_loop, args=(self._message_queue,))
#         loop_process.start()

#     def module_loop(self, queue: Queue):
#         img_name_list = listdir(self.mock_images_path)

#         for img_name in img_name_list:
#             img_path = self.mock_images_path + '/' + img_name

#             img = PILOpen(img_path)
#             self.mock_images.append(img)

#         while True:

#             sleep(2)
#             index = random.randint(0, len(self.mock_images) - 1)

#             message = (
#                 "hand_detected",
#                 self.mock_images[index]
#             )

#             queue.put(message)
