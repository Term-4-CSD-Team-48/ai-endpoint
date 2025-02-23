import time
from sam2.build_sam import build_sam2_camera_predictor
import torch
import numpy as np
import cv2


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# build model
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# obtain video source
cap = cv2.VideoCapture("IMG_4210.mov")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
             cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Set the mouse callback function for the window
def click_event(event, x, y, flags, param):
    global points
    global labels
    global first_frame
    global predictor
    if event == cv2.EVENT_LBUTTONDOWN:
        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        first_frame = True
        predictor.reset_state()


points = np.array([[0, 0]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_event)

# write video
out = cv2.VideoWriter("sam2.1.avi",
                      cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

first_frame = True
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if first_frame:
        first_frame = False
        predictor.load_first_frame(frame)

        ann_frame_idx = 0  # the frame index we interact with
        # give a unique id to each object we interact with (it can be any integers)
        ann_obj_id = 1
        # Let's add a positive click at (x, y) = (210, 350) to get started

        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[width / 2, height / 2]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )

        # ! add bbox
        # bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        # )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)
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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    out.write(frame)
    end_time = time.time()
    duration = end_time - start_time

    frame_skip_interval = int(fps * duration)
    for i in range(frame_skip_interval):
        cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# garbage collection
out.release()
cap.release()
cv2.destroyAllWindows()
