import cv2
from queue import Queue
import time
from threading import Thread


class Streamer:
    def __init__(self, url):
        print("Creating Streamer object for", url)
        self.cap = cv2.VideoCapture(url)
        self.url = url
        self.Q = Queue(maxsize=2)
        self.running = True
        print("Streamer object created for", url)

    def info(self):
        print("==============================Stream Info==============================")
        print("| Stream:", self.url, "|")
        print("| Queue Size:", self.Q.qsize(), "|")
        print("| Running:", self.running, "|")
        print("======================================================================")

    def get_processed_frame(self):
        if self.Q.empty():
            return None
        return self.Q.queue[0]

    def release(self):
        """
        Release the Streamer.
        """
        self.cap.release()

    def stop(self):
        """
        Stop the Streamer.
        """
        print("Stopping", self.cap, "Status", self.url)
        self.running = False

    def start(self):
        """
        Start the Streamer.
        """
        print("Starting streamer", self.cap, "Status", self.running)
        while self.running:
            ret, frame = self.cap.read()
            # print(frame,ret)
            if not ret:
                print("NO Frame for", self.url)
                continue
            # exit()
            if not self.Q.full():
                print("Streamer PUT", self.Q.qsize())
                self.Q.put({"frame": frame, "time": time.time()})
                print("Streamer PUT END", self.Q.qsize())
        self.release()


if __name__ == "__main__":
    streamer = Streamer("rtsp://localhost:8554/105")

    thread = Thread(target=streamer.start)
    thread.start()

    while streamer.running:
        data = streamer.get_processed_frame()
        if data is None:
            continue
        frame = data["frame"]
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
