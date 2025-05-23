import numpy as np
import torch
import cv2
import requests
import os

from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Tracker:
    def __init__(self):
        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self._sam = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self._points = np.array([[0, 0]], dtype=np.float32)
        self.labels = np.array([1], dtype=np.int32)
        self.changed_points = False
        self._observer_ip = ""
        self._observer_port = os.environ.get("API_PORT", "8000")
        self._object_on_screen = True

    def prompt_first_frame(self, frame):
        points = self.points
        labels = self.labels
        self._sam.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self._sam.add_new_prompt(
            frame_idx=0, obj_id=1, points=points, labels=labels
        )
        return self._draw_masks_and_points_on_frame(frame, out_obj_ids, out_mask_logits, points)

    def track(self, frame):
        points = self.points
        labels = self.labels
        out_obj_ids, out_mask_logits = None, None
        if not self.changed_points:
            out_obj_ids, out_mask_logits = self._sam.track(frame)
        else:
            self.changed_points = False
            self._sam.load_first_frame(frame)
            self._sam.reset_state()
            _, out_obj_ids, out_mask_logits = self._sam.add_new_prompt(
                frame_idx=0, obj_id=1, points=points, labels=labels
            )
        return self._draw_masks_and_points_on_frame(frame, out_obj_ids, out_mask_logits, points)

    def _draw_masks_and_points_on_frame(self, frame, out_obj_ids, out_mask_logits, points):
        # Masks
        height, width = frame.shape[:2]
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        # Points
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

        # Check if object is on screen and trigger _on_update
        object_on_screen = bool(np.any(out_mask))
        self._on_update(object_on_screen)

        return frame, object_on_screen

    def _on_update(self, object_on_screen):
        if self._object_on_screen != object_on_screen:
            print("updating observer " + self.observer_ip)
            self._object_on_screen = object_on_screen
            data = {"objectOnScreen": object_on_screen}
            if self.observer_ip is not None and len(self.observer_ip) > 0:
                url = f"http://{self.observer_ip}:{self.observer_port}/ai/on-update"
                try:
                    requests.post(url, json=data)
                except Exception as e:
                    print(f"WARNING: an error occured in sending request to {url}")

    def get_points(self):
        return self._points

    def set_points(self, points):
        """
        Call reset_state immediately after this
        """
        print("SamTracker points have changed")
        self.changed_points = True
        self._points = points

    def get_observer_ip(self):
        return self._observer_ip

    def set_observer_ip(self, ip):
        print("setting observer ip " + ip)
        self._observer_ip = ip

    def get_observer_port(self):
        return self._observer_port

    def set_observer_port(self, port):
        print("setting observer port " + port)
        self._observer_port = port

    points = property(get_points, set_points)
    observer_ip = property(get_observer_ip, set_observer_ip)
    observer_port = property(get_observer_port, set_observer_port)


class BGRToTrackerAdapter:
    def __init__(self, tracker: Tracker):
        self.tracker = tracker

    def _bgr_frame_to_rgb_frame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def prompt_first_frame(self, frame):
        return self.tracker.prompt_first_frame(self._bgr_frame_to_rgb_frame(frame))

    def track(self, frame):
        return self.tracker.track(self._bgr_frame_to_rgb_frame(frame))
