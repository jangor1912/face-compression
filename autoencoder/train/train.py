from matplotlib import pyplot

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.small import VariationalAutoEncoder, LSTMEncoder32, LSTMDecoder32
from dataset.batch_generator import BatchGenerator


class Training(object):
    def __init__(self, model, training_generator, validation_generator, metric):
        self.model = model
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.metric = metric

    def train(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=[self.metric])
        # train model
        history = self.model.fit_generator(generator=self.training_generator.get_batch(),
                                           epochs=1,
                                           validation_data=self.validation_generator.get_batch(),
                                           steps_per_epoch=len(self.training_generator),
                                           validation_steps=len(self.validation_generator),
                                           verbose=2)
        # plot metrics
        pyplot.plot(history.history['mse'])
        pyplot.show()


def train_small():
    train_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/train/final"
    test_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test/final"
    batch_size = 4
    frames_no = 8
    input_shape = (32, 32, 3)
    latent_size = 512
    train_gen = BatchGenerator(train_directory, input_size=input_shape[:-1],
                               batch_size=batch_size, frames_no=frames_no)
    test_gen = BatchGenerator(test_directory, input_size=input_shape[:-1],
                              batch_size=batch_size, frames_no=frames_no)
    metric = FaceMetric.get_loss_from_batch
    encoder = LSTMEncoder32(inputShape=input_shape, batchSize=batch_size, framesNo=frames_no,
                            latentSize=latent_size, latentConstraints='bvae', beta=100., capacity=512.,
                            randomSample=True)
    decoder = LSTMDecoder32(inputShape=input_shape, batchSize=batch_size, latentSize=latent_size)
    auto_encoder = VariationalAutoEncoder(encoder, decoder)
    auto_encoder.summary()
    t = Training(model=auto_encoder.model,
                 training_generator=train_gen,
                 validation_generator=test_gen,
                 metric=metric)
    t.train()


if __name__ == "__main__":
    train_small()
