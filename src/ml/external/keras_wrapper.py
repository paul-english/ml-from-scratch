import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.objectives import mean_absolute_error
from keras.optimizers import Adam, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor


class WrappedKerasRegressor(object):

    @staticmethod
    def parameters():
        return {
            'epochs': [50],
        }

    def __init__(self, epochs=500):
        self.epochs = epochs

    def fit(self, X, y):
        def build_model():
            model = Sequential()
            model.add(Dense(128, input_dim=X.shape[1], init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, init='uniform', activation='linear'))

            adam = Adam()

            model.compile(loss='mse', optimizer=adam, metrics=['mae', 'mse'])
            return model

        self.model = KerasRegressor(build_model, nb_epoch=self.epochs, batch_size=1024, verbose=1)
        self._history = self.model.fit(X, y)

        # TODO do this outside of the thing so we can name this better
        plt.figure()
        plt.plot(self._history.history['mean_squared_error'])
        plt.savefig("visualizations/keras_mse.png")

        plt.figure()
        plt.plot(self._history.history['mean_absolute_error'])
        plt.savefig("visualizations/keras_mae.png")

    def predict(self, X):
        y_hat = self.model.predict(X)
        print('---- yhat', y_hat)
        return y_hat
