from pathlib import Path

from matplotlib import pyplot
from tensorflow.python.keras.utils import OrderedEnqueuer

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.small import VariationalAutoEncoder, LSTMEncoder32, LSTMDecoder32
from dataset.batch_generator import BatchSequence
from diagnoser.diagnoser import ModelDiagonoser


class Training(object):
    def __init__(self, model, training_sequence, validation_sequence, metric, callbacks):
        self.model = model
        self.training_sequence = training_sequence
        self.validation_sequence = validation_sequence
        self.metric = metric
        self.callbacks = callbacks

    def train(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=[self.metric])

        # create ordered queues
        train_enqueuer = OrderedEnqueuer(self.training_sequence, use_multiprocessing=True, shuffle=True)
        train_enqueuer.start(workers=4, max_queue_size=24)
        train_gen = train_enqueuer.get()
        test_enqueuer = OrderedEnqueuer(self.validation_sequence, use_multiprocessing=True, shuffle=True)
        test_enqueuer.start(workers=4, max_queue_size=24)
        test_gen = test_enqueuer.get()
        # train model
        history = self.model.fit_generator(generator=train_gen,
                                           epochs=100,
                                           validation_data=next(test_gen),
                                           verbose=2,
                                           steps_per_epoch=len(self.training_sequence),
                                           validation_steps=len(self.validation_sequence),
                                           callbacks=self.callbacks)
        # plot metrics
        print(str(history.history))
        # pyplot.plot(history.history['get_loss_from_batch'])
        # pyplot.show()


def train_small():
    train_directory = Path("G:/Magisterka/kaggle_dataset/small/train/final")
    test_directory = Path("G:/Magisterka/kaggle_dataset/small/test/final")
    batch_size = 1
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
                            latentSize=latent_size, latentConstraints='bvae', beta=100., capacity=512.,
                            randomSample=True)
    decoder = LSTMDecoder32(inputShape=input_shape, batchSize=batch_size, latentSize=latent_size)
    auto_encoder = VariationalAutoEncoder(encoder, decoder)
    auto_encoder.summary()
    t = Training(model=auto_encoder.model,
                 training_sequence=train_seq,
                 validation_sequence=test_seq,
                 metric=metric,
                 callbacks=callbacks)
    t.train()


if __name__ == "__main__":
    train_small()
