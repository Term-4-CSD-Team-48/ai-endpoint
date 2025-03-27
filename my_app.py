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
EXT_X_TARGETDURATION = 6
EXT_X_VERSION = 6
M3U8_FILE_HEADER_FORMAT = "#EXTM3U\n#EXT-X-VERSION:{EXT_X_VERSION}\n#EXT-X-TARGETDURATION:{EXT_X_TARGETDURATION}\n#EXT-X-MEDIA-SEQUENCE:{EXT_X_MEDIA_SEQUENCE}\n#EXT-X-INDEPENDENT-SEGMENTS"
M3U8_FILE_HEADER = M3U8_FILE_HEADER_FORMAT.format(
    EXT_X_VERSION=EXT_X_VERSION, EXT_X_TARGETDURATION=EXT_X_TARGETDURATION, EXT_X_MEDIA_SEQUENCE=0)

segments = []
segment = []
segment_duration = 0
threshold_segment_duration = EXT_X_TARGETDURATION - 2
connected_to_RTMP_server = False


def create_app():
    app = Flask(__name__)

    def reset_m3u8():
        # Reset .m3u8
        with open(M3U8_FILE, "w") as f:
            # #EXT-X-ENDLIST\n
            f.write(M3U8_FILE_HEADER)
        print("Resetted .m3u8 file with " + M3U8_FILE_HEADER)
        # Delete all .ts files
        count = 0
        for filename in os.listdir(HLS_DIR):
            if filename.endswith(".ts"):
                file_path = os.path.join(HLS_DIR, filename)
                try:
                    os.remove(file_path)
                    count = count + 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        print(f"Remove {count} .ts files")

    # Thread
    def get_and_process_frames():
        global connected_to_RTMP_server
        global segment
        global segment_duration
        global sam
        while True:
            print("Connecting to RTMP server...")
            streamer = Streamer("rtmp://127.0.0.1/live/stream")

            if not streamer.isOpened():
                connected_to_RTMP_server = False
                print("Error: Cannot open RTMP stream. Retrying in 5 seconds")
                time.sleep(5)  # Wait before retrying
                continue

            # Initialize
            connected_to_RTMP_server = True
            print("Connected to RTMP server!")
            time.sleep(1)  # Required to relinquish control to streamer thread to get first frame
            reset_m3u8()
            ret, previous_frame = streamer.read()  # previous_frame is an np.array
            t1 = time.time()  # Start time of previous_frame
            if not ret:
                raise Exception("Failed to get first frame!")

            # Sam processes first frame and draw mask and point on it
            height, width = previous_frame.shape[:2]
            print(f"Width: {width}, Height: {height}")
            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)  # SAM processes in RGB
            previous_frame, object_on_screen = sam.prompt_first_frame(previous_frame)

            while True:
                ret, current_frame = streamer.read()
                t2 = time.time()  # End time of previous_frame / Start time of current_frame

                if not ret:
                    print(
                        "Warning: Failed to retrieve frame. Reconnecting after 3 seconds delay")
                    connected_to_RTMP_server = False
                    break

                # Process current_frame with SAM and draw mask and point on it
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)  # SAM processes in RGB
                current_frame, object_on_screen = sam.track(current_frame)

                # Turn previous_frame to bytes for ffmpeg processing
                _, previous_frame = cv2.imencode('.jpg', previous_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                previous_frame = previous_frame.tobytes()
                frame_time = t2 - t1
                segment.append((previous_frame, frame_time))
                segment_duration = segment_duration + frame_time

                # Set previous_frame to current_frame and t1 to t2
                previous_frame = current_frame
                t1 = t2
                time.sleep(1/64)

            streamer.release()
            time.sleep(3)

    # Thread
    def segment_to_ts():
        global connected_to_RTMP_server
        global segment
        global segment_duration
        global threshold_segment_duration
        segment_filename_idx = 0
        ffmpeg_process = None
        print("segment_to_ts thread initialized")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-g", "30",
            "-r", "30",
            "-hls_time", f"{EXT_X_TARGETDURATION}",
            "-hls_list_size", "0",
            "-hls_flags", "append_list",
            os.path.join(HLS_DIR, "stream.m3u8")
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        while True:
            if not connected_to_RTMP_server:
                print("segment_to_ts_thread going to sleep now for 5s as not connected to RTMP server")
                time.sleep(5)
                continue

            for frame_bytes, _ in segment:
                print("processing 1 frame")
                ffmpeg_process.stdin.write(frame_bytes)
            segment_duration = 0
            segment.clear()

            # Convert segment to ts file once enough time has lapsed
            if segment_duration >= threshold_segment_duration:
                # Output file name and path
                output_filename = f"{segment_filename_idx}.ts"
                segment_filename_idx += 1
                output_path = os.path.join(HLS_DIR, output_filename)

                # Use FFMPEG to write the .ts file
                # `ffmpeg` command to convert raw frames to a .ts file with H.264 codec
                fps = len(segment)/segment_duration
                # TODO: Figure out how to make independent segments if .ts file still failing
                # -fflags +genpts
                if ffmpeg_process is None:
                    command = [
                        'ffmpeg',
                        '-y',  # Overwrite output file without asking
                        '-f', 'image2pipe',
                        '-fflags', '+genpts',  # generate presentation timestamps (PTS)
                        '-vcodec', 'mjpeg',
                        '-framerate', str(fps),  # Set dynamic frame rate
                        '-i', '-',  # Input comes from stdin
                        '-c:v', 'libx264',  # Use H.264 codec
                        '-preset', 'ultrafast',  # Fastest encoding (use 'medium' for better compression)
                        '-r', '30',  # Output is constant 30 fps
                        '-f', 'mpegts',  # Output format .ts
                        output_path
                    ]
                    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)

                # Write each frame in the batch to ffmpeg's stdin and create the ts file
                for frame_bytes, duration in segment:
                    ffmpeg_process.stdin.write(frame_bytes)
                    ffmpeg_process.stdin.flush()
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                # Update .m3u8 playlist
                update_m3u8(output_filename, segment_duration)

                # Post-op cleanup
                segment.clear()
                segment_duration = 0
                # ffmpeg_process = None
            time.sleep(1/4)

    def update_m3u8(filename: str, segment_duration):
        # with open(M3U8_FILE, "r+") as file:
        #     # Move the pointer (similar to a cursor in a text editor) to the end of the file
        #     file.seek(0, os.SEEK_END)

        #     # This code means the following code skips the very last character in the file -
        #     # i.e. in the case the last line is null we delete the last line
        #     # and the penultimate one
        #     pos = file.tell() - 1

        #     # Read each character in the file one at a time from the penultimate
        #     # character going backwards, searching for a newline character
        #     # If we find a new line, exit the search
        #     while pos > 0 and file.read(1) != "\n":
        #         pos -= 1
        #         file.seek(pos, os.SEEK_SET)

        #     # So long as we're not at the start of the file, delete all the characters ahead
        #     # of this position
        #     if pos > 0:
        #         file.seek(pos, os.SEEK_SET)
        #         file.truncate()
        #         file.write(
        #             f"\n#EXTINF:{round(segment_duration, 6)},\n{filename}\n"
        #         )
        # global segments
        # segments.append((filename, segment_duration))
        # if (len(segments) > 5):
        #     segments.pop(0)

        # body = ""
        # for f, dur in segments:
        #     body = body + f"\n#EXTINF:{round(dur, 6)},\n{f}"

        # with open(M3U8_FILE, "w") as file:
        #     file.write(
        #         M3U8_FILE_HEADER_FORMAT.format(EXT_X_VERSION=EXT_X_VERSION,
        #                                        EXT_X_TARGETDURATION=EXT_X_TARGETDURATION, EXT_X_MEDIA_SEQUENCE=segments[0][0].split('.')[0]) + body
        #     )
        with open(M3U8_FILE, "a") as file:
            file.write(
                f"\n#EXTINF:{round(segment_duration, 6)},\n{filename}"
            )
        print("Updated m3u8")

    get_and_process_frames_thread = threading.Thread(target=get_and_process_frames, daemon=True)
    get_and_process_frames_thread.start()
    segment_to_ts_thread = threading.Thread(target=segment_to_ts, daemon=True)
    segment_to_ts_thread.start()
    return app


app = create_app()


@app.route('/ping', methods=['GET'])
def ping():
    print(f"Received request at /ping from {request.headers.get('X-Forwarded-For', request.remote_addr)}")
    return "healthy" if torch.cuda.is_available() and sam else "unhealthy"


@app.route('/invocations', methods=['POST'])
def invocations():
    global sam
    data = request.get_json()  # Extract JSON data from the request
    client_ip = request.headers.get('X-Forward-For', request.remote_addr)
    print(f"Received request at /invocations from {client_ip} with {data}")
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


if __name__ == '__main__':
    app.run(port=8080)
