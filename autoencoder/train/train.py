import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import OrderedEnqueuer

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.small import VariationalAutoEncoder, LSTMEncoder32, LSTMDecoder32
from dataset.batch_generator import BatchSequence
from diagnoser.diagnoser import ModelDiagonoser


class Training(object):
    def __init__(self, model, training_sequence, validation_sequence, metric, callbacks, output_dir):
        self.model = model
        self.training_sequence = training_sequence
        self.validation_sequence = validation_sequence
        self.metric = metric
        self.callbacks = callbacks
        self.output_dir = output_dir

    def train(self):
        self.model.compile(loss=self.metric, optimizer='adam', metrics=[self.metric, "mse", "mae"])

        # create ordered queues
        train_enqueuer = OrderedEnqueuer(self.training_sequence, use_multiprocessing=True, shuffle=True)
        train_enqueuer.start(workers=4, max_queue_size=24)
        train_gen = train_enqueuer.get()
        test_enqueuer = OrderedEnqueuer(self.validation_sequence, use_multiprocessing=True, shuffle=True)
        test_enqueuer.start(workers=4, max_queue_size=24)
        test_gen = test_enqueuer.get()
        # train model
        history = self.model.fit_generator(generator=train_gen,
                                           epochs=250,
                                           validation_data=next(test_gen),
                                           verbose=2,
                                           steps_per_epoch=len(self.training_sequence),
                                           validation_steps=len(self.validation_sequence),
                                           callbacks=self.callbacks)
        # self.model.save(str(Path(self.output_dir, 'model.h5')))
        # plot metrics
        self.plot_results(history, self.output_dir)

    @staticmethod
    def plot_results(history, output_path):
        history = history.history
        print(history)
        y_loss = [np.average(seq) for seq in history["loss"]]
        y_val_loss = [np.average(seq) for seq in history["val_loss"]]
        epochs = [i for i in range(len(y_loss))]

        plt.figure()
        plt.plot(epochs, history["get_loss_from_batch"], color="blue", label="training")
        plt.plot(epochs, history["val_get_loss_from_batch"], color="red", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Face-metric value")
        plt.legend(["training", "validation"], loc='upper left')
        plt.title('Face-metric')
        # plt.show()
        plt.savefig(str(Path(output_path, 'face-metric.png')))
        plt.close()
        np.savetxt(str(Path(output_path, 'face-metric-train.csv')), history["get_loss_from_batch"], delimiter=",")
        np.savetxt(str(Path(output_path, 'face-metric-validation.csv')),
                   history["val_get_loss_from_batch"], delimiter=",")

        plt.figure()
        plt.plot(epochs, history["mean_squared_error"], color="blue", label="training")
        plt.plot(epochs, history["val_mean_squared_error"], color="red", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("MSE value")
        plt.legend(["training", "validation"], loc='upper left')
        plt.title('MSE metric')
        # plt.show()
        plt.savefig(str(Path(output_path, 'mse-metric.png')))
        plt.close()
        np.savetxt(str(Path(output_path, 'mse-metric.csv')), history["mean_squared_error"], delimiter=",")
        np.savetxt(str(Path(output_path, 'mse-metric-validation.csv')),
                   history["val_mean_squared_error"], delimiter=",")

        plt.figure()
        plt.plot(epochs, history["mean_absolute_error"], color="blue", label="training")
        plt.plot(epochs, history["val_mean_absolute_error"], color="red", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("MAE value")
        plt.legend(["training", "validation"], loc='upper left')
        plt.title('MAE metric')
        # plt.show()
        plt.savefig(str(Path(output_path, 'mae-metric.png')))
        plt.close()
        np.savetxt(str(Path(output_path, 'mae-metric.csv')), history["mean_absolute_error"], delimiter=",")
        np.savetxt(str(Path(output_path, 'mae-metric-validation.csv')),
                   history["val_mean_absolute_error"], delimiter=",")

        plt.figure()
        plt.plot(epochs, y_loss, color="blue", label="training")
        plt.plot(epochs, y_val_loss, color="red", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss value")
        plt.legend(["training", "validation"], loc='upper left')
        plt.title('Model loss')
        # plt.show()
        plt.savefig(str(Path(output_path, 'model-loss.png')))
        plt.close()
        np.savetxt(str(Path(output_path, 'model-loss.csv')), y_loss, delimiter=",")
        np.savetxt(str(Path(output_path, 'model-loss-validation.csv')),
                   y_val_loss, delimiter=",")


def train_small():
    train_directory = Path("G:/Magisterka/kaggle_dataset/small/train/no_blur")
    test_directory = Path("G:/Magisterka/kaggle_dataset/small/test/no_blur")
    batch_size = 8
    frames_no = 8
    input_shape = (32, 32, 3)
    latent_size = 512

    train_seq = BatchSequence(train_directory, input_size=input_shape[:-1],
                              batch_size=batch_size, frames_no=frames_no)
    test_seq = BatchSequence(test_directory, input_size=input_shape[:-1],
                             batch_size=batch_size, frames_no=frames_no)

    num_samples = 3  # samples to be generated each epoch
    samples_dir = Path("G:/Magisterka/kaggle_dataset/small/results/samples")
    callbacks = [ModelDiagonoser(test_seq, batch_size, num_samples, samples_dir)]

    metric = FaceMetric.get_loss_from_batch
    encoder = LSTMEncoder32(inputShape=input_shape, batchSize=batch_size, framesNo=frames_no,
                            latentSize=latent_size, latentConstraints='bvae', alpha=50., beta=10., capacity=512.,
                            randomSample=True)
    decoder = LSTMDecoder32(inputShape=input_shape, batchSize=batch_size, latentSize=latent_size)
    auto_encoder = VariationalAutoEncoder(encoder, decoder)
    auto_encoder.summary()
    t = Training(model=auto_encoder.model,
                 training_sequence=train_seq,
                 validation_sequence=test_seq,
                 metric=metric,
                 callbacks=callbacks,
                 output_dir=samples_dir)
    t.train()


if __name__ == "__main__":
    train_small()
