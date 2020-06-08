import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.callbacks import Callback
# Depending on your keras version:-
# from tensorflow.python.keras.engine.training import GeneratorEnqueuer, Sequence, OrderedEnqueuer
from tensorflow.python.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer

from dataset.batch_generator import BatchSequence


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    image = BatchSequence.np_img_to_rgb_image(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.FileWriter(self.outdir,
                                            flush_secs=10)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
                                global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagonoser(Callback):

    def __init__(self,
                 data_generator,
                 batch_size,
                 num_samples,
                 output_dir,
                 tensorboard=True):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.tensorboard_writer = TensorBoardWriter(output_dir)
        self.data_generator = data_generator
        self.tensorboard = tensorboard
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=False,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=False,
                                              wait_time=0.01)
        self.enqueuer.start(workers=1, max_queue_size=32)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            generator_output = next(output_generator)
            x, y = generator_output[:2]
            y_pred = self.model.predict(x)[0]
            y_true = y[0]

            for i in range(0, len(y_pred)):
                n = steps_done * self.batch_size + i
                if n >= self.num_samples:
                    return

                pred = y_pred[i]

                ground_truth = y_true[i]

                if self.tensorboard:
                    self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                       .format(epoch, sample_index), ground_truth)
                    self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                       .format(epoch, sample_index), pred)
                else:
                    pass
                sample_index += 1

            steps_done += 1

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
