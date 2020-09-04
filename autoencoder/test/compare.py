import tensorflow as tf
import numpy as np

from common import utils
from video.deconstructor.deconstructor import Deconstructor
from skimage.metrics import structural_similarity as ssim


def calculate_differences(original_video, compressed_video, csv_path):
    original_video_generator = Deconstructor.video_to_images(original_video)
    compressed_video_generator = Deconstructor.video_to_images(compressed_video)
    for original_frame, compressed_frame in zip(original_video_generator, compressed_video_generator):
        psnr_val = tf.image.psnr(original_frame, compressed_frame, max_val=255)
        ssim_val = ssim(original_frame, compressed_frame, max_val=255)

        mse_val = np.mean(np.abs(original_frame - compressed_frame), axis=-1)
        mae_val = np.mean(np.square(original_frame - compressed_frame), axis=-1)
        utils.append_csv_row(csv_path, ["psnr", "ssim", "mse", "mae"],
                             {"psnr": psnr_val,
                              "ssim": ssim_val,
                              "mae": mae_val,
                              "mse": mse_val})


def main():
    pass


if __name__ == "__main__":
    main()
