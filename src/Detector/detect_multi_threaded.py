from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
import multiprocessing
import queue
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
from threading import Thread
import numpy as np



# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


def worker(input_q, output_q, box_q, cap_params, frame_processed):
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
            # output_q.put(frame)
            output_q.put(frame)
            frame_processed += 1

        else:
            output_q.put(frame)

        if box_image is not None and box_image.size != 0:
            box_q.put(box_image)
        else:
            box_q.put(blank_image)
    sess.close()

class Detector:
    def __init__(self, args, score_thresh):
        self.args = args
        self.input_q = multiprocessing.Queue(maxsize=self.args.queue_size)
        self.output_q = multiprocessing.Queue(maxsize=self.args.queue_size)
        self.box_q = multiprocessing.Queue(maxsize=self.args.queue_size)

        self.video_capture = WebcamVideoStream(
            src=self.args.video_source, width=self.args.width, height=self.args.height).start()

        cap_params = {}
        frame_processed = 0
        cap_params['im_width'], cap_params['im_height'] = self.video_capture.size()
        cap_params['score_thresh'] = score_thresh

        # max number of hands we want to detect/track
        cap_params['num_hands_detect'] = self.args.num_hands

        print(cap_params, self.args)

        # spin up workers to paralleize detection.
        self.pool = multiprocessing.Pool(self.args.num_workers, worker,
                    (self.input_q, self.output_q, self.box_q, cap_params, frame_processed))

        self.start_time = datetime.datetime.now()
        self.num_frames = 0
        self.fps = 0
        self.index = 0
        self.frame = None
        self.box_image = None
        self.blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                print("Detector stop")
                return
            try:
                frame = self.video_capture.read()
                frame = cv2.flip(frame, 1)
                self.index += 1
            except:
                print("video read exception")

            try:
                self.input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                output_frame = self.output_q.get()
            except:
                print("Queue excpetion")
            
            try:
                self.box_image = self.box_q.get(False)
                self.box_image = cv2.cvtColor(self.box_image, cv2.COLOR_RGB2BGR)
            except queue.Empty:
                self.box_image = self.blank_image

            try:
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            except:
                print("cvtColor exception")

            try:
                self.elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
                self.num_frames += 1
                self.fps = self.num_frames / self.elapsed_time
                # print("frame ",  index, num_frames, elapsed_time, fps)
                self.frame = output_frame
            except:
                print("calculations excpetion")


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

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300, #1024
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200, #768
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=2,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    score_thresh = 0.5

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Box', cv2.WINDOW_NORMAL)
    detector = Detector(args, score_thresh).start()

    try:
        while True:
            output_frame = detector.getFrame()
            fps = detector.getFps()
            if (output_frame is not None):
                    if (args.fps > 0):
                        detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                        output_frame)
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                        
                    box_image = detector.getBoxImage()
                    # if box_image is not None:
                    cv2.imshow('Box', box_image)
                    

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # print("video end")
                # break
                pass
    # except cv2.error as e:
    except Exception as e:
        print("ERROR - ", str(e))
    
    detector.stop()
    print("koniec")
    
