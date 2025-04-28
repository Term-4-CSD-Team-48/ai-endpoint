# 50.001 AI server

## Getting started

### Running the server

If you are on linux, run

./serve.sh

If you are on windows, you will not be able to run the server directly. You will need to build the image directly and run it in a container. The image is already available for pulling with the below command

docker pull public.ecr.aws/l1g2c1s4/50.001:ai

The server should be run with

docker run --gpus all -p 8080:8080 -p 1935:1935 -e API_PORT=80 public.ecr.aws/l1g2c1s4/50.001:ai

This server uses GPU, listens on ports 8080 HTTP and 1935 RTMP. If your machine
has no GPU the HTTP server will crash.

The server will think that the API's port is located at 80 through the -e API_PORT=80 option but you have to change it if you're hosting the API on a different port. There's no need to tell the AI server the API's ip as it will just take the ip address of the last GET /observe request.

### Streaming the livefeed to the AI server for processing

You can stream to the AI server with a camera (1) or a file (2). The steps assume that the AI server is hosted on same machine so 127.0.0.1. It might run on a separate machine on the same network too like 192.168.1.6 for example that's fine too. You'll need to install ffmpeg and add it to PATH.

1. To stream using your camera follow these steps first.

ffmpeg -list_devices true -f dshow -i dummy

Something like [dshow @ 000001f4f2426bc0] "Integrated Camera" (video) will show up. The names
might differ.

Run this command to start streaming to the RTMP server.

ffmpeg -f dshow -rtbufsize 100M -pixel_format yuyv422 -i video="Integrated Camera" -c:v libx264 -s 640x360 -pix_fmt yuv420p -bufsize 1200k -b:v 600k -preset ultrafast -tune zerolatency -f flv rtmp://127.0.0.1/live/stream

2. Alternatively, you can stream from a file instead like this

ffmpeg -i input.mp4 -c:v libx264 -s 640x360 -pix_fmt yuv420p -bufsize 1200k -b:v 600k -preset ultrafast -tune zerolatency -f flv rtmp://127.0.0.1/live/stream

### Changing the AI server's port

Not possible if you're on windows as you need to reconfigure nginx and you'd need to run docker build which takes a lot of time.

## Issues

### AI server not being able to update API when object is not in screen

This is an issue related to windows. The server is developed to natively run on linux unfortunately as we intended it to run on an AWS EC2 instance (cloud server). This is because the AI is very taxing on the GPU and we do not want to bring our desktops with better GPU to school. The reason why the AI server is not able to update the API is because the container that the server is running inside on is has its own network bridge that obfuscates the original IP of the request. As such the AI is unable to extract the API's IP address for future reference and will not be able to talk with the API. That being said, strangely it's able to return responses so there should be a way to fix this but I do not have the time and I need to prepare for my finals.

### API receiving 403 responses from POST /prompt and POST /observe

The AI will return 403 when the request doesn't come from a local ip address. Please ensure your're hosting the API and AI on the same LAN. The alternative is to modify the code to not reject requests that are not from the above local addresses.

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
