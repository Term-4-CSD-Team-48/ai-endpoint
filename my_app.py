import torch
import numpy as np
import threading
import cv2
from io import BytesIO
from PIL import Image
import subprocess
import base64
import json
import atexit
from flask import Flask, request, jsonify
import os
import time
from queue import Queue

from streamer import Streamer
from sam_tracker import SamTracker
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build model
sam = SamTracker()

HLS_DIR = "/mnt/hls"
M3U8_FILE = os.path.join(HLS_DIR, "stream.m3u8")

observer_id = None


def create_app():
    app = Flask(__name__)

    # Thread
    def process_to_hls():
        global sam
        while True:
            print("Connecting to RTMP server...")
            streamer = Streamer("rtmp://127.0.0.1/live/stream")

            if not streamer.isOpened():
                print("Error: Cannot open RTMP stream. Retrying in 5 seconds")
                time.sleep(5)  # Wait before retrying
                continue

            # Initialize
            print("Connected to RTMP server!")
            time.sleep(1)  # Required to relinquish control to streamer thread to get first frame

            # Set up FFmpeg command to stream processed frames to RTMP
            ffmpeg_stream_processed_command = [
                'ffmpeg',
                '-re',
                '-f', 'rawvideo',  # Raw video format (no container)
                '-s', '640x360',  # Input resolution
                '-pixel_format', 'bgr24',
                '-i', '-',  # Input from stdin (pipe)
                '-r', '10',
                '-pix_fmt', 'yuv420p',
                '-c:v', 'libx264',  # Video codec (H.264)
                '-bufsize', '64M',
                '-maxrate', '4M',
                '-f', 'flv',  # Output format for streaming
                'rtmp://127.0.0.1/live/processed',  # RTMP URL
            ]

            # Set up FFmpeg command to convert processed frames to HLS
            ffmpeg_processed_to_hls_command = [
                'ffmpeg',
                '-re',
                '-i', 'rtmp://127.0.0.1/live/processed',
                '-c:v', 'libx264',
                '-crf', '26',  # 51 is worst 1 is best
                '-preset', 'ultrafast',
                '-g', '10',
                '-sc_threshold', '0',
                '-f', 'hls',
                '-hls_time', '4',
                '-hls_flags', 'independent_segments',
                '-hls_playlist_type', 'event',
                M3U8_FILE
            ]

            # Start the FFmpeg processes
            ffmpeg_stream_processed_process = subprocess.Popen(ffmpeg_stream_processed_command, stdin=subprocess.PIPE)
            ffmpeg_processed_to_hls_process = subprocess.Popen(ffmpeg_processed_to_hls_command)

            ret, previous_frame = streamer.read()  # previous_frame is an np.array
            if not ret:
                raise Exception("Failed to get first frame!")

            # Sam processes first frame and draw mask and point on it
            height, width = previous_frame.shape[:2]
            print(f"Width: {width}, Height: {height}")
            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)  # SAM processes in RGB
            previous_frame, object_on_screen = sam.prompt_first_frame(previous_frame)

            while True:
                # Get latest frame
                ret, current_frame = streamer.read()

                if not ret:
                    print("Warning: Failed to retrieve frame exiting loop")
                    break

                # Process current_frame with SAM and draw mask and point on it
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)  # SAM processes in RGB
                current_frame, object_on_screen = sam.track(current_frame)

                # Turn previous_frame to bytes for ffmpeg processing
                ffmpeg_stream_processed_process.stdin.write(previous_frame)

                # Set previous_frame to current_frame and t1 to t2
                previous_frame = current_frame
                time.sleep(1/64)

            print("Not processing frames sleeping for 3s")
            streamer.release()
            ffmpeg_stream_processed_process.stdin.close()  # Close stdin pipe
            ffmpeg_stream_processed_process.wait()
            ffmpeg_processed_to_hls_process.stdin.close()
            ffmpeg_processed_to_hls_process.wait()
            time.sleep(3)

    thread = threading.Thread(target=process_to_hls, daemon=True)
    thread.start()
    return app


app = create_app()


@app.route('/ping', methods=['GET'])
def ping():
    print(f"Received request at /ping from {request.headers.get('X-Forwarded-For', request.remote_addr)}")
    return "healthy" if torch.cuda.is_available() and sam else "unhealthy"


@app.route('/prompt', methods=['POST'])
def prompt():
    global sam
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

    sam.points = np.array([[data['x'], data['y']]], dtype=np.float32)

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
    global observer_id
    print(f"Received request at /observe")
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"Received request at /observe from {client_ip}")
    if not client_ip.startswith("10.0"):
        return "outsiders not allowed", 403
    data = request.get_json()
    if data['jSessionId'] is None:
        return "no owner id", 400
    sam._observer_ip = client_ip
    observer_id = data['jSessionId']
    return "ok"


if __name__ == '__main__':
    app.run(port=8080)
