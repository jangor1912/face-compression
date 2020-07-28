import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.optimizers import Adamax
from tensorflow.python.keras.utils import OrderedEnqueuer
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.big import VariationalAutoEncoder128
from autoencoder.models.big_lstm import VariationalLSTMAutoEncoder128
from autoencoder.models.custom_nvae import NVAEAutoEncoder128
from autoencoder.models.small import VariationalAutoEncoder, LSTMEncoder32, LSTMDecoder32
from dataset.batch_generator import BatchSequence, LSTMSequence, NVAESequence
from callbacks.callbacks import ModelDiagonoser, KLWeightScheduler, LearningRateSchedulerPerBatch, DotDict


class Training(object):
    def __init__(self, model, training_sequence, validation_sequence,
                 metric, metrics,
                 callbacks, output_dir,
                 epochs=100,
                 compile_model=True):
        self.model = model
        self.training_sequence = training_sequence
        self.validation_sequence = validation_sequence
        self.metric = metric
        self.metrics = metrics or [self.metric, "mse", "mae"]
        self.callbacks = callbacks
        self.output_dir = output_dir
        self.epochs = epochs
        self.compile_model = compile_model

    def train(self):
        if self.compile_model:
            optimizer = Adamax(lr=0.008)
            self.model.compile(loss=self.metric, optimizer=optimizer, metrics=self.metrics)

        # create ordered queues
        train_enqueuer = OrderedEnqueuer(self.training_sequence, use_multiprocessing=False, shuffle=True)
        train_enqueuer.start(workers=1, max_queue_size=64)
        train_gen = train_enqueuer.get()
        test_enqueuer = OrderedEnqueuer(self.validation_sequence, use_multiprocessing=False, shuffle=True)
        test_enqueuer.start(workers=1, max_queue_size=64)
        test_gen = test_enqueuer.get()
        # train model
        history = self.model.fit_generator(generator=train_gen,
                                           epochs=self.epochs,
                                           validation_data=next(test_gen),
                                           verbose=2,
                                           steps_per_epoch=len(self.training_sequence),
                                           validation_steps=len(self.validation_sequence),
                                           callbacks=self.callbacks)
        # plot metrics
        self.plot_results(history, self.output_dir)
        self.model.save_weights(str(Path(self.output_dir, 'model.h5')))

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


def get_default_hparams():
    """ Return default hyper-parameters """
    params_dict = {
        # Experiment Params:
        'is_training': True,  # train mode (relevant only for accelerated LSTM mode)
        'epochs': 50,  # how many times to go over the full train set (on average, since batches are drawn randomly)
        'save_every': None,  # Batches between checkpoints creation and validation set evaluation.
        # Once an epoch if None.
        'batch_size': 100,  # Minibatch size. Recommend leaving at 100.
        # Loss Params:
        'optimizer': 'adam',  # adam or sgd
        'learning_rate': 0.001,
        'decay_rate': 0.9999,  # Learning rate decay per minibatch.
        'min_learning_rate': .00001,  # Minimum learning rate.
        'kl_tolerance': 0.2,  # Level of KL loss at which to stop optimizing for KL.
        'kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
        'kl_weight_start': 1e-4,  # KL start weight when annealing.
        'kl_decay_rate': 0.9995,  # KL annealing decay rate per minibatch.
        'mask_kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
        'mask_kl_weight_start': 0.001,  # KL start weight when annealing.
        'mask_kl_decay_rate': 0.995,  # KL annealing decay rate per minibatch.
        'grad_clip': 1.0,  # Gradient clipping. Recommend leaving at 1.0.
        "gamma": 1.0,  # Parameter to boost face metric
    }

    return params_dict


def get_callbacks_dict(auto_encoder, model_params,
                       test_seq, batch_size,
                       num_samples, samples_directory):
    """ create a dictionary of all used callbacks """

    # Callbacks dictionary
    callbacks_dict = dict()

    # Checkpoints callback
    checkpoints_dir = Path(samples_directory, 'checkpoints')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_file = Path(checkpoints_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    callbacks_dict['model_checkpoint'] = ModelCheckpoint(filepath=str(checkpoints_file),
                                                         monitor='val_get_loss_from_batch',
                                                         save_best_only=True, mode='min')

    # KL loss weight decay callback, custom callback
    callbacks_dict['kl_weight_schedule'] = KLWeightScheduler(
        schedule=lambda step: (model_params.kl_weight -
                               (model_params.kl_weight - model_params.kl_weight_start) *
                               model_params.kl_decay_rate ** step),
        kl_weight=auto_encoder.kl_weight,
        name="Face encoder KL weight",
        verbose=1)
    callbacks_dict['mask_kl_weight_schedule'] = KLWeightScheduler(
        schedule=lambda step: (model_params.mask_kl_weight -
                               (model_params.mask_kl_weight - model_params.mask_kl_weight_start) *
                               model_params.mask_kl_decay_rate ** step),
        kl_weight=auto_encoder.mask_kl_weight,
        name="Mask encoder KL weight",
        verbose=1)

    # LR decay callback, modified to apply decay each batch as in original implementation
    # callbacks_dict['lr_schedule'] = LearningRateSchedulerPerBatch(
    #     lambda step: ((model_params.learning_rate - model_params.min_learning_rate) * model_params.decay_rate ** step
    #                   + model_params.min_learning_rate))

    callbacks_dict['model_diagnoser'] = ModelDiagonoser(test_seq, batch_size, num_samples, samples_directory)

    return callbacks_dict


def train_small(train_directory, test_directory, samples_directory, epochs=100):
    batch_size = 8
    frames_no = 8
    input_shape = (32, 32, 3)
    latent_size = 512

    train_seq = BatchSequence(train_directory, input_size=input_shape[:-1],
                              batch_size=batch_size, frames_no=frames_no)
    test_seq = BatchSequence(test_directory, input_size=input_shape[:-1],
                             batch_size=batch_size, frames_no=frames_no)

    num_samples = 3  # samples to be generated each epoch
    callbacks = [ModelDiagonoser(test_seq, batch_size, num_samples, samples_directory)]

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
                 metrics=None,
                 callbacks=callbacks,
                 output_dir=samples_directory,
                 epochs=epochs)
    t.train()


def train_big(train_directory, test_directory, samples_directory,
              epochs=100,
              model=None):
    alpha = 10.0
    beta = 10.0
    batch_size = 8
    frames_no = 16
    input_shape = (128, 128, 3)

    train_seq = BatchSequence(train_directory, input_size=input_shape[:-1],
                              batch_size=batch_size, frames_no=frames_no)
    test_seq = BatchSequence(test_directory, input_size=input_shape[:-1],
                             batch_size=batch_size, frames_no=frames_no)

    num_samples = 3  # samples to be generated each epoch
    callbacks = [ModelDiagonoser(test_seq, batch_size, num_samples, samples_directory)]

    metric = FaceMetric.get_loss_from_batch
    if not model:
        auto_encoder = VariationalAutoEncoder128(batch_size=batch_size,
                                                 alpha=alpha,
                                                 beta=beta)
        auto_encoder.summary()
        model = auto_encoder.model
    t = Training(model=model,
                 training_sequence=train_seq,
                 validation_sequence=test_seq,
                 metric=metric,
                 metrics=None,
                 callbacks=callbacks,
                 output_dir=samples_directory,
                 epochs=epochs)
    t.train()


def train_lstm(train_directory, test_directory, samples_directory,
               epochs=100,
               model=None,
               alpha=0.01,
               beta=0.01,
               gamma=10.0,
               batch_size=4):
    decoder_frames_no = 3
    encoder_frames_no = 30
    input_shape = (128, 128, 3)

    train_seq = LSTMSequence(train_directory, input_size=input_shape[:-1],
                             batch_size=batch_size, frames_no=decoder_frames_no,
                             encoder_frames_no=encoder_frames_no)
    test_seq = LSTMSequence(test_directory, input_size=input_shape[:-1],
                            batch_size=batch_size, frames_no=decoder_frames_no,
                            encoder_frames_no=encoder_frames_no)

    num_samples = 3  # samples to be generated each epoch
    callbacks = [ModelDiagonoser(test_seq, batch_size, num_samples, samples_directory)]

    metric = FaceMetric(None, gamma=gamma).get_loss_from_batch
    if not model:
        auto_encoder = VariationalLSTMAutoEncoder128(batch_size=batch_size,
                                                     encoder_frames_no=encoder_frames_no,
                                                     decoder_frames_no=decoder_frames_no,
                                                     alpha=alpha,
                                                     beta=beta)
        auto_encoder.summary()
        model = auto_encoder.model
    t = Training(model=model,
                 training_sequence=train_seq,
                 validation_sequence=test_seq,
                 metric=metric,
                 metrics=None,
                 callbacks=callbacks,
                 output_dir=samples_directory,
                 epochs=epochs)
    t.train()


def train_vae(train_directory, test_directory, samples_directory,
              epochs=100,
              batch_size=4,
              encoder_frames_no=30,
              initial_epoch=0,
              checkpoint_path=None):
    encoder_frames_no = encoder_frames_no
    input_shape = (128, 128, 3)

    train_seq = NVAESequence(train_directory, input_size=input_shape[:-1],
                             batch_size=batch_size,
                             encoder_frames_no=encoder_frames_no)
    test_seq = NVAESequence(test_directory, input_size=input_shape[:-1],
                            batch_size=batch_size,
                            encoder_frames_no=encoder_frames_no)

    num_samples = 3  # samples to be generated each epoch
    model_params = DotDict(get_default_hparams())

    auto_encoder = NVAEAutoEncoder128(model_params,
                                      batch_size=batch_size,
                                      encoder_frames_no=encoder_frames_no)
    auto_encoder.summary()
    model = auto_encoder.model
    metric = auto_encoder.loss_func
    metrics = [auto_encoder.loss_func,
               auto_encoder.face_metric,
               auto_encoder.face_kl_loss,
               auto_encoder.mask_kl_loss,
               "mae", "mse"]
    optimizer = Adamax(lr=0.008)
    model.compile(loss=metric, optimizer=optimizer, metrics=metrics)

    callbacks_dict = get_callbacks_dict(auto_encoder, model_params,
                                        test_seq, batch_size,
                                        num_samples, samples_directory)

    if checkpoint_path is not None:
        # Load weights:
        model.load_trained_weights(checkpoint_path)
        # Initial batch (affects LR and KL weight decay):
        num_batches = len(train_seq)
        count = initial_epoch * num_batches
        callbacks_dict['lr_schedule'].count = count
        callbacks_dict['kl_weight_schedule'].count = count
        callbacks_dict['mask_kl_weight_schedule'].count = count

    callbacks = [callback for callback in callbacks_dict.values()]

    t = Training(model=model,
                 training_sequence=train_seq,
                 validation_sequence=test_seq,
                 metric=metric,
                 metrics=metrics,
                 callbacks=callbacks,
                 output_dir=samples_directory,
                 epochs=epochs,
                 compile_model=False)
    t.train()


if __name__ == "__main__":
    train_directory = Path("G:/Magisterka/youtube_dataset/output/cleared/train")
    test_directory = Path("G:/Magisterka/youtube_dataset/output/cleared/test")
    samples_dir = Path("G:/Magisterka/youtube_dataset/output/cleared/samples")
    train_vae(train_directory, test_directory, samples_dir, epochs=10, batch_size=1)
