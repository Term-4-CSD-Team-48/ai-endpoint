import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import json
from flask import Flask, request

from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# build model
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
points = np.array([[0, 0]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
first_frame = True

app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
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
