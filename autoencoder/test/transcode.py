from pathlib import Path

import re

import os
from os.path import isfile, join

import subprocess


def transcode(input_path, output_path, bit_rate=2500):
    bash_command = ["ffmpeg", "-i", f"{input_path}",
                    "-c:v", "libx264", "-b:v", f"{str(bit_rate)}K",
                    "-maxrate", f"{str(bit_rate)}K", "-bufsize", f"{str(bit_rate/2)}K",
                    f"{output_path}"]
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f"Output = {output}")
    print(f"Errors = {error}")


def yield_videos(videos_dir):
    files = {join(videos_dir, f) for f in os.listdir(videos_dir) if isfile(join(videos_dir, f))}
    for file_path in files:
        if re.search(r'.*-processed-video\d.avi', file_path):
            org_video_path = file_path[:file_path.find(".avi")]
            yield org_video_path


if __name__ == "__main__":
    input_dir = "/home/gorazda/Desktop/videos/test"
    output_dir = "/home/gorazda/Desktop/videos/800kbps"
    for input_video_path in yield_videos(input_dir):
        output_video_path = str(Path(output_dir, input_video_path.split("/")[-1]))
        kbps = 800  # kb
        transcode(input_video_path + ".avi", output_video_path + ".mp4",
                  bit_rate=kbps)
