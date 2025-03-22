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
from flask import Flask, request
import os
import time
from queue import Queue

from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# build model
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
points = np.array([[0, 0]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
first_frame = True


HLS_DIR = "/mnt/hls"
M3U8_FILE = os.path.join(HLS_DIR, "stream.m3u8")
EXT_X_TARGETDURATION = 6
segment = []
threshold_segment_duration = EXT_X_TARGETDURATION - 2
connected_to_RTMP_server = False


def create_app():
    app = Flask(__name__)

    def get_and_process_frames():
        global connected_to_RTMP_server
        global segment
        while True:
            print("Connecting to RTMP server...")
            cap = cv2.VideoCapture("rtmp://127.0.0.1/live/stream")

            if not cap.isOpened():
                connected_to_RTMP_server = False
                print("Error: Cannot open RTMP stream. Retrying in 5 seconds")
                time.sleep(5)  # Wait before retrying
                continue

            # Initialize
            connected_to_RTMP_server = True
            print("Connected to RTMP server!")
            reset_m3u8()
            ret, previous_frame = cap.read()
            t0 = time.time()  # Start time of segment
            t1 = t0  # Start time of previous_frame
            if not ret:
                raise Exception("Failed to get first frame!")
            else:
                print("Finished first frame read")

            while True:
                ret, current_frame = cap.read()
                t2 = time.time()  # End time of previous_frame / Start time of frame / Start time of next segment
                t1 = t2
                if not ret:
                    print(
                        "Warning: Failed to retrieve frame. Reconnecting after 3 seconds delay")
                    connected_to_RTMP_server = False
                    break
                print("Obtained current_frame")
                _, previous_frame = cv2.imencode('.jpg', previous_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                previous_frame = previous_frame.tobytes()
                segment.append((previous_frame, t2 - t1))
                previous_frame = current_frame
                time.sleep(0.33333333333)  # 30 fps

            cap.release()
            time.sleep(3)

    def reset_m3u8():
        print("Resetting .m3u8 playlist...")
        # Reset .m3u8
        with open(M3U8_FILE, "w") as f:
            f.write(
                f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:{EXT_X_TARGETDURATION}\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-ENDLIST")
        print("Resetted .m3u8 file")
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

    def update_m3u8(filename: str, segment_duration):
        print("Updating m3u8")
        with open(M3U8_FILE, "r+") as file:
            # Move the pointer (similar to a cursor in a text editor) to the end of the file
            file.seek(0, os.SEEK_END)

            # This code means the following code skips the very last character in the file -
            # i.e. in the case the last line is null we delete the last line
            # and the penultimate one
            pos = file.tell() - 1

            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                file.seek(pos, os.SEEK_SET)

            # So long as we're not at the start of the file, delete all the characters ahead
            # of this position
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()
                file.write(
                    f"#EXTINF:{round(segment_duration, 6)},\n{filename}\n#EXT-X-DISCONTINUITY\n#EXT-X-ENDLIST"
                )
        print("Updated m3u8")

    def segment_to_ts():
        global connected_to_RTMP_server
        global segment
        global threshold_segment_duration
        segment_filename_idx = 0
        old_segment_length = 0
        segment_duration = 0
        ffmpeg_process = None
        print("segment_to_ts thread initialized")
        while True:
            if not connected_to_RTMP_server:
                print("segment_to_ts_thread going to sleep now for 1s")
                time.sleep(1)
                continue
            print("Checking if segment duration exceeds threshold")
            new_segment_length = len(segment)
            if new_segment_length != old_segment_length:
                difference = new_segment_length - old_segment_length
                for i in range(old_segment_length, new_segment_length, difference):
                    segment_duration = segment_duration + segment[i][1]
                old_segment_length = new_segment_length
                print(f"Segment duration is at {segment_duration}s")
            else:
                print(f"new_segment_length: {new_segment_length}, old_segment_length: {old_segment_length}")
                time.sleep(0.33333333333)  # 30 fps
            time.sleep(0.33333333333)  # 30 fps

            # Convert segment to ts file once enough time has lapsed
            if segment_duration >= threshold_segment_duration:
                print("Converting segment to ts " + segment_duration + "s")
                # Output file name and path
                output_filename = f"segment_{segment_filename_idx}.ts"
                segment_filename_idx += 1
                output_path = os.path.join(HLS_DIR, output_filename)

                # Use FFMPEG to write the .ts file
                # `ffmpeg` command to convert raw frames to a .ts file with H.264 codec
                if ffmpeg_process is None:
                    command = [
                        'ffmpeg',
                        '-y',  # Overwrite output file without asking
                        '-f', 'image2pipe',
                        '-vcodec', 'mjpeg',
                        '-r', '30',
                        '-i', '-',  # Input comes from stdin
                        '-c:v', 'libx264',  # Use H.264 codec
                        '-preset', 'ultrafast',  # Fastest encoding (use 'medium' for better compression)
                        '-vsync', 'vfr',  # Allow variable frame rate
                        '-f', 'mpegts',  # Output format .ts
                        output_path
                    ]
                    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)

                # Write each frame in the batch to ffmpeg's stdin and create the ts file
                for frame_bytes, _ in segment:
                    ffmpeg_process.stdin.write(frame_bytes)
                    ffmpeg_process.stdin.flush()
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                # Update .m3u8 playlist
                update_m3u8(output_filename, segment_duration)

                # Post-op cleanup
                segment.clear()
                old_segment_length = 0
                segment_duration = 0
                ffmpeg_process = None

    get_and_process_frames_thread = threading.Thread(target=get_and_process_frames, daemon=True)
    get_and_process_frames_thread.start()
    segment_to_ts_thread = threading.Thread(target=segment_to_ts, daemon=True)
    segment_to_ts_thread.start()
    return app


app = create_app()


@app.route('/ping', methods=['GET'])
def ping():
    print('pinged')
    return "healthy" if torch.cuda.is_available() and predictor else "unhealthy"


@app.route('/invocations', methods=['POST'])
def invocations():
    global first_frame
    global points

    # Get request data
    file = request.files['file']
    prompt = request.form.get('prompt')

    # Read the image file
    frame = Image.open(BytesIO(file.read()))
    frame = np.array(frame)
    height, width = frame.shape[:2]

    if prompt is not None:
        # Assuming prompt is a string representation of a dictionary
        prompt_data = eval(prompt)
        points = np.array(
            [[prompt_data['x'], prompt_data['y']]], dtype=np.float32)
        predictor.load_first_frame(frame)
        if not first_frame:
            predictor.reset_state()
        first_frame = False
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=1, points=points, labels=labels
        )
    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

    # Draw masks on the frame
    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(0, len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
            np.uint8
        ) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

    # Draw points on the frame
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

    # Convert back to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check if object is on screen
    object_on_screen = bool(np.any(out_mask))

    # Convert frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Create JSON response
    response = {
        "object_on_screen": object_on_screen,
        "frame": frame_base64
    }

    return json.dumps(response)


if __name__ == '__main__':
    app.run(port=8080)
