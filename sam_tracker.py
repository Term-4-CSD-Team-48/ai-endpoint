import numpy as np
import torch
import cv2

from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class SamTracker:
    def __init__(self):
        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self._sam = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self._points = np.array([[0, 0]], dtype=np.float32)
        self.labels = np.array([1], dtype=np.int32)
        self.changed_points = False

    def prompt_first_frame(self, frame):
        points = self.points
        labels = self.labels
        self._sam.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self._sam.add_new_prompt(
            frame_idx=0, obj_id=1, points=points, labels=labels
        )
        return self.draw_masks_and_points_on_frame(frame, out_obj_ids, out_mask_logits, points)

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
        return self.draw_masks_and_points_on_frame(frame, out_obj_ids, out_mask_logits, points)

    def draw_masks_and_points_on_frame(self, frame, out_obj_ids, out_mask_logits, points):
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

        # Convert back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if object is on screen
        object_on_screen = bool(np.any(out_mask))

        return frame, object_on_screen

    def get_points(self):
        return self._points

    def set_points(self, points):
        """
        Call reset_state immediately after this
        """
        print("SamTracker points have changed")
        self.changed_points = True
        self._points = points

    points = property(get_points, set_points)
