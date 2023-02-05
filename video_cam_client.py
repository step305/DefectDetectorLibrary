import socket
import base64
import numpy as np
import cv2
import json
import time
import cv2
import multiprocessing as mp

HOST, PORT = "192.168.136.131", 8008


def receive_packet(sock):
    chunk = ''
    data_list = []
    while '\r\n' not in chunk:
        chunk = sock.recv(1024).decode()
        data_list.append(chunk)
    data = ''.join(data_list)
    pack = json.loads(data)
    return pack, len(data)


def run_loop(cam, queue_out, stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.settimeout(1)
    time.sleep(1)
    s.send(b'Hello')
    packet, _ = receive_packet(s)
    buf_decode = base64.b64decode(packet['image'])
    timestamp0 = packet['timestamp']
    print('Camera started')
    while not stop.is_set():
        try:
            s.send(b'get')
            packet, size_data = receive_packet(s)
            buf_decode = base64.b64decode(packet['image'])
            timestamp = (packet['timestamp'] - timestamp0) / 1e6
            jpg = np.frombuffer(buf_decode, np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)
            if not queue_out.full():
                queue_out.put(cv2.resize(frame, (1280, 720)))
        except Exception:
            break
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
