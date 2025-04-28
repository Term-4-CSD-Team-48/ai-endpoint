import os
import subprocess


class RGBFramesToHLSProcess:
    def __init__(self):
        HLS_DIR = "/mnt/hls"
        os.makedirs('/mnt/hls', exist_ok=True)
        M3U8_FILE = os.path.join(HLS_DIR, "stream.m3u8")
        process_command = [
            'ffmpeg',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'rawVideo',
            '-s', '640x360',
            '-pixel_format', 'rgb24',
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-g', '25',  # default output fps is 25
            '-crf', '26',  # 51 is worst 1 is best
            '-preset', 'ultrafast',
            '-sc_threshold', '0',
            '-f', 'hls',
            '-hls_time', '4',
            '-hls_flags', 'independent_segments',
            '-hls_playlist_type', 'event',
            M3U8_FILE
        ]
        self.process = subprocess.Popen(process_command, stdin=subprocess.PIPE)

    def write(self, bytes):
        self.process.stdin.write(bytes)

    def release(self):
        self.process.stdin.close()
        self.process.wait()

# # Set up FFmpeg command to convert processed frames to HLS
    # ffmpeg_processed_to_hls_command = [
    #     'ffmpeg',
    #     '-re',
    #     '-i', 'rtmp://127.0.0.1/live/processed',
    #     '-c:v', 'libx264',
    #     '-crf', '26',  # 51 is worst 1 is best
    #     '-preset', 'ultrafast',
    #     '-g', '10',
    #     '-sc_threshold', '0',
    #     '-f', 'hls',
    #     '-hls_time', '4',
    #     '-hls_flags', 'independent_segments',
    #     '-hls_playlist_type', 'event',
    #     M3U8_FILE
    # ]

# # Set up FFmpeg command to stream processed frames to RTMP
    # ffmpeg_stream_processed_command = [
    #     'ffmpeg',
    #     '-re',
    #     '-f', 'rawvideo',  # Raw video format (no container)
    #     '-s', '640x360',  # Input resolution
    #     '-pixel_format', 'bgr24',
    #     '-i', '-',  # Input from stdin (pipe)
    #     '-r', '10',
    #     '-pix_fmt', 'yuv420p',
    #     '-c:v', 'libx264',  # Video codec (H.264)
    #     '-bufsize', '64M',
    #     '-maxrate', '4M',
    #     '-f', 'flv',  # Output format for streaming
    #     'rtmp://127.0.0.1/live/processed',  # RTMP URL
    # ]
