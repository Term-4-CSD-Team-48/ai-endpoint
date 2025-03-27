import cv2
from queue import Queue
import time
from threading import Thread


class Streamer:
    def __init__(self, url):
        self._cap = cv2.VideoCapture(url)
        if self._cap.isOpened():
            self._thread = Thread(target=self._update, daemon=True)
            self._thread.start()
            self._opened = True
            print("Streamer thread listening for ", url)
        else:
            self._opened = False
            print("Failed to start a thread to listen for ", url)

    def _update(self):
        while True:
            if self._cap.isOpened():
                (self._ret, self._frame) = self._cap.read()
            time.sleep(1/128)

    def isOpened(self):
        return self._opened

    def release(self):
        self._cap.release()

    def read(self):
        return self._ret, self._frame
