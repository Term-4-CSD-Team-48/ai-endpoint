import torch
import numpy as np
import threading
import cv2
import subprocess
from flask import Flask, request, jsonify
import os
import time

from streamer import Streamer
from tracker import Tracker, BGRToTrackerAdapter
from processes import RGBFramesToHLSProcess

tracker = Tracker()
bgr_to_tracker_adapter = BGRToTrackerAdapter(tracker)

observer_id = None


def create_app():
    app = Flask(__name__)

    # Thread
    def rtmp_to_ai_to_hls_thread():
        global tracker
        while True:
            streamer = Streamer("rtmp://127.0.0.1/live/stream")

            if not streamer.isOpened():
                print("Error: Cannot open RTMP stream. Retrying in 5 seconds")
                time.sleep(5)  # Wait before retrying
                continue

            # Initialize
            print("Connected to RTMP server!")
            time.sleep(1)  # Required to relinquish control to streamer thread to get first frame

            # Start the FFmpeg process
            process = RGBFramesToHLSProcess()

            ret, previous_frame = streamer.read()  # previous_frame is an np.array
            if not ret:
                raise Exception("Failed to get first frame!")

            # Sam processes first frame and draw mask and point on it
            previous_frame, _ = bgr_to_tracker_adapter.prompt_first_frame(previous_frame)

            while True:
                # Get latest frame
                ret, current_frame = streamer.read()

                if not ret:
                    print("Warning: Failed to retrieve frame exiting loop")
                    break

                # Process current_frame with SAM and draw mask and point on it
                current_frame, _ = bgr_to_tracker_adapter.track(current_frame)

                # Turn previous_frame to bytes for ffmpeg processing
                process.write(previous_frame.tobytes())

                # Set previous_frame to current_frame
                previous_frame = current_frame
                time.sleep(1/64)

            streamer.release()
            process.release()
            print("Not processing frames sleeping for 3s")
            time.sleep(3)

    thread = threading.Thread(target=rtmp_to_ai_to_hls_thread, daemon=True)
    thread.start()
    return app


app = create_app()


@app.route('/ping', methods=['GET'])
def ping():
    print(f"Received request at /ping from {request.headers.get('X-Forwarded-For', request.remote_addr)}")
    return "healthy" if torch.cuda.is_available() and tracker else "unhealthy"


@app.route('/prompt', methods=['POST'])
def prompt():
    global tracker
    print(f"Received request at /prompt")
    data = request.get_json()  # Extract JSON data from the request
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"Received request at /prompt from {client_ip} with {data}")
    if not client_ip.startswith("10.0"):
        return "outsiders not allowed", 403
    if not data['x'] or not data['y']:
        return "invalid JSON", 400  # Handle case where JSON is missing

    # Extract a cookie from the request
    # cookie_value = request.cookies.get('JSESSIONID', None)
    # if cookie_value is None:
    #     return "unauthorized", 401

    tracker.points = np.array([[data['x'], data['y']]], dtype=np.float32)

    return "changed points", 200


@app.route('/auth_request', methods=['GET'])
def auth_request():
    print('Received request at /auth_request')
    global observer_id
    cookie_value = request.cookies.get('JSESSIONID', None)
    return "ok"
    if cookie_value is None:
        return "no owner id", 401
    if cookie_value != observer_id:
        return "forbidden", 403
    return "ok"


@app.route('/observe', methods=['POST'])
def observe():
    # global observer_id
    # print(f"Received request at /observe")
    # client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    # print(f"Received request at /observe from {client_ip}")
    # if not client_ip.startswith("10.0"):
    #     return "outsiders not allowed", 403
    # data = request.get_json()
    # if data['jSessionId'] is None:
    #     return "no owner id", 400
    # tracker.observer_ip = client_ip
    # observer_id = data['jSessionId']
    return "ok"


if __name__ == '__main__':
    app.run(port=8080)
