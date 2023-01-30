import time

import cv2
import multiprocessing as mp


def run_loop(cam, queue_out, stop):
    print('Camera started')
    # cap = cv2.VideoCapture(cam)
    cap = cv2.VideoCapture('test.mp4')
    # frame = cv2.imread('dataset\\detection_verification\\test\\test.jpg')
    # frame = cv2.resize(frame, (1920, 1080))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap.set(cv2.CAP_PROP_FPS, 30)

    while not stop.is_set():
        ret, frame = cap.read()
        time.sleep(0.2)
        # ret, _ = cap.read()
        if ret:
            if not queue_out.full():
                queue_out.put(cv2.resize(frame, (1280, 720)))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap. release()
    print('Camera stopped')


class VideoCamera:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_buf = mp.Queue(10)
        self.stop_event = mp.Event()
        self.cam_proc = mp.Process(target=run_loop, args=(self.cam_id, self.frame_buf, self.stop_event))
        self.cam_proc.start()

    def get(self):
        if self.stop_event.is_set():
            return None
        if not self.frame_buf.empty():
            return self.frame_buf.get()
        else:
            return None

    def stop(self):
        self.stop_event.set()
        self.cam_proc.terminate()
        print('Camera terminated')
