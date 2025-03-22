import torch
import numpy as np
import threading
import cv2
from io import BytesIO
from PIL import Image
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

# Frames for hls
frame_queue = Queue()
HLS_DIR = "/mnt/hls"
M3U8_FILE = os.path.join(HLS_DIR, "stream.m3u8")

# Maintain a list of segments
segment_list = []
max_segments = 5  # Keep only the last 5 segments to prevent storage overflow


def create_app():
    app = Flask(__name__)

    def get_frames():
        global frame_queue
        while True:
            print("Connecting to RTMP server...")
            cap = cv2.VideoCapture("rtmp://127.0.0.1/live/stream")

            if not cap.isOpened():
                print("Error: Cannot open RTMP stream. Retrying in 5 seconds")
                time.sleep(5)  # Wait before retrying
                continue

            print("Connected to RTMP server!")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(
                        "Warning: Failed to retrieve frame. Reconnecting after 3 seconds delay")
                    break

                # Convert frame to NumPy array (it already is, but this ensures compatibility)
                frame_array = np.array(frame)
                frame_queue.put(frame_array)

            cap.release()
            cv2.destroyAllWindows()
            time.sleep(3)

    def update_m3u8():
        print("Updating .m3u8 playlist...")
        with open(M3U8_FILE, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write("#EXT-X-TARGETDURATION:4\n")  # Approximate segment length
            f.write("#EXT-X-MEDIA-SEQUENCE:{}\n".format(len(segment_list) -
                    max_segments if len(segment_list) > max_segments else 0))

            for segment in segment_list[-max_segments:]:
                f.write("#EXTINF:4.0,\n")
                f.write(f"{segment}\n")

    def process_frames():
        global first_frame
        global points
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()

                # Generate a unique filename
                timestamp = int(time.time() * 1000)
                output_filename = f"frame_{timestamp}.ts"
                output_path = os.path.join(HLS_DIR, output_filename)

                # Encode as a short .ts segment (simulating HLS chunk)
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc,
                                      30.0, (frame.shape[1], frame.shape[0]))
                out.write(frame)
                out.release()

                # Update segment list and remove old segments
                segment_list.append(output_filename)
                if len(segment_list) > max_segments:
                    old_segment = segment_list.pop(0)
                    # Delete old .ts file
                    os.remove(os.path.join(HLS_DIR, old_segment))

                # Update .m3u8 playlist
                update_m3u8()
                print(f"Processed and saved frame to {output_path}")
            time.sleep(0.03333333333)  # 30 FPS

    model_thread = threading.Thread(target=get_frames, daemon=True)
    model_thread.start()
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
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
    app.run(port=8080, debug=True)
