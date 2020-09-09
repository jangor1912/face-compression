import os
import re
import subprocess
from os.path import isfile, join
from pathlib import Path


def transcode(input_path, output_path, bit_rate=2500, compression="libx264"):
    bash_command = ["ffmpeg", "-i", f"{input_path}",
                    "-vf", "scale=128:128",
                    "-c:v", compression, "-b:v", f"{str(bit_rate)}K",
                    "-maxrate", f"{str(bit_rate)}K", "-bufsize", f"{str(bit_rate / 2)}K",
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
    input_dir = "/home/jan/Desktop/compression/test"
    output_dir = "/home/jan/Desktop/compression/h265"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for kbps in [800, 400, 200, 100, 50]:
        cur_output_dir = Path(output_dir, f"{kbps}kbps")
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        for input_video_path in yield_videos(input_dir):
            output_video_path = str(Path(cur_output_dir, input_video_path.split("/")[-1]))
            transcode(input_video_path + ".avi", output_video_path + ".mp4",
                      bit_rate=kbps, compression="libx265")
