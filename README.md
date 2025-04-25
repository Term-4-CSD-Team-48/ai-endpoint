# 50.001 AI server

## Getting started

### Running the server

This server has been containerized and is available for pulling with

docker pull public.ecr.aws/l1g2c1s4/50.001:ai

The server should be run with

docker run --gpus all -p 8080:8080 -p 1935:1935 public.ecr.aws/l1g2c1s4/50.001:ai

This server listens on ports 1935 RTMP and 8080 HTTP and uses GPU. If your machine
has no GPU the HTTP server will crash.

### Streaming the livefeed to the AI server for processing

First obtain the video device names on your machine with

ffmpeg -list_devices true -f dshow -i dummy

Something like [dshow @ 000001f4f2426bc0] "Integrated Camera" (video) will show up. The names
might differ.

Run this command to start streaming to the RTMP server.

ffmpeg -f dshow -rtbufsize 100M -pixel_format yuyv422 -i video="Integrated Camera" -c:v libx264 -s 640x360 -pix_fmt yuv420p -bufsize 1200k -b:v 600k -preset ultrafast -tune zerolatency -f flv rtmp://127.0.0.1/live/stream

I'm assuming that the AI server is being hosted on the same machine as the streaming machine
so the IP is 127.0.0.1

## Issues

### API receiving 403 responses from POST /prompt and POST /observe

The AI will return 403 when the request doesn't come from 192.168.0.0/16 or 10.0.0.0/16 or localhost
Please ensure your're hosting the API and AI on the same LAN. The alternative is to
modify the code to not reject requests that are not from the above local addresses.

# segment-anything-2 real-time

Run Segment Anything Model 2 on a **live video stream**

## News

- 13/12/2024 : Update to sam2.1
- 20/08/2024 : Fix management of `non_cond_frame_outputs` for better performance and add bbox prompt

## Demos

<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>

</div>

## Getting Started

### Installation

```bash
pip install -e .
```

### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Camera prediction

```python
import torch
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture(<your video or camera >)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(<your promot >)

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            ...
```

### With model compilation

You can use the `vos_inference` argument in the `build_sam2_camera_predictor` function to enable model compilation. The inference may be slow for the first few execution as the model gets warmed up, but should result in significant inference speed improvement.

We provide the modified config file `sam2/configs/sam2.1/sam2.1_hiera_t_512.yaml`, with the modifications necessary to run SAM2 at a 512x512 resolution. Notably the parameters that need to be changed are highlighted in the config file at lines 24, 43, 54 and 89.

We provide the file `sam2/benchmark.py` to test the speed gain from using the model compilation.

## References:

- SAM2 Repository: https://github.com/facebookresearch/sam2
