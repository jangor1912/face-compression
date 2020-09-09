from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

from common import utils
from video.deconstructor.deconstructor import Deconstructor


def calculate_differences(original_video, compressed_video, csv_path):
    original_video_generator = Deconstructor.video_to_images(original_video)
    compressed_video_generator = Deconstructor.video_to_images(compressed_video)
    for original_frame, compressed_frame in zip(original_video_generator, compressed_video_generator):
        original_frame = original_frame[0]
        compressed_frame = compressed_frame[0]

        mae_val = np.mean(np.abs(original_frame - compressed_frame))
        mse_val = np.mean(np.square(original_frame - compressed_frame))
        ssim_val = ssim(original_frame, compressed_frame,
                        max_val=255,
                        multichannel=True)

        original_frame = tf.image.convert_image_dtype(original_frame, tf.float32)
        compressed_frame = tf.image.convert_image_dtype(compressed_frame, tf.float32)

        psnr_val = tf.image.psnr(original_frame, compressed_frame, max_val=255)
        utils.append_csv_row(csv_path, ["psnr", "ssim", "mse", "mae"],
                             {"psnr": float(psnr_val),
                              "ssim": float(ssim_val),
                              "mae": mae_val,
                              "mse": mse_val})
        # print(f"MSE = {mse_val}")
        # print(f"MAE = {mae_val}")
        # print(f"PSNR = {psnr_val}")
        # print(f"SSIM = {ssim_val}")


def compare_videos(original_dir: Path, compressed_dir: Path, results_file: Path,
                   suffix: str = ".mp4"):
    for file in original_dir.iterdir():
        print(f"Processing video {file.name}")
        original_video = file
        compressed_video = str(Path(compressed_dir, file.name))
        # change suffix
        compressed_video = compressed_video[:compressed_video.rfind(".")]
        compressed_video += suffix
        calculate_differences(str(original_video), compressed_video, results_file)


def compare_compressions(original_dir: Path, main_dir: Path):
    for compression in ["h264", "h265"]:
        for bit_rate in [800, 400, 200, 100, 50, 25]:
            print(f"Comparing compression {compression} with bit-rate {bit_rate}kbps")
            bit_rate = str(bit_rate) + "kbps"
            compressed_dir = Path(main_dir, compression, bit_rate)
            results_file = Path(main_dir, compression + "-" + bit_rate + ".csv")
            compare_videos(original_dir, compressed_dir, results_file)


def main():
    original_dir = Path("/home/jan/Desktop/compression/8000kbps")
    compressed_dir = Path("/home/jan/Desktop/compression/face_decoded")
    compare_videos(original_dir, compressed_dir, Path("/home/jan/Desktop/compression/face_decoded.csv"),
                   suffix=".avi")
    # main_dir = Path("/home/jan/Desktop/compression")
    # compare_compressions(original_dir, main_dir)


if __name__ == "__main__":
    main()
