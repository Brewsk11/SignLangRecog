from utils import detect_multi_threaded
from utils import detector_utils as detector_utils
import queue
import cv2

if __name__ == '__main__':

    score_thresh = 0.5

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Box', cv2.WINDOW_NORMAL)
    detector = detect_multi_threaded.Detector(score_thresh).start()

    try:
        while True:
            output_frame = detector.getFrame()
            fps = detector.getFps()
            if (output_frame is not None):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), output_frame)
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                        
                    box_image = detector.getBoxImage()
                    # if box_image is not None:
                    cv2.imshow('Box', box_image)

                    except_table = detector.getExceptTable()
                    for item in except_table:
                        print(item)

                    detector.clearExceptTable()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # print("video end")
                # break
                pass
    except cv2.error as e:
    # except Exception as e:
        print("ERROR - ", str(e))

    detector.stop()