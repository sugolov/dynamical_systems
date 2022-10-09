import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from polynomial import Approximator


def Lorenz(dt, x, b=8 / 3, sig=10, r=28):
    return (
        x + dt * np.array([
            sig * (x[1] - x[0]),
            r * x[0] - x[0] * x[2] - x[1],
            x[0] * x[1] - b * x[2]
        ])
    )


def compile_model(lr=0.01, width=10, depth=3):
    model = tfk.models.Sequential()

    model.add(Dense(width,
                    activation="sigmoid",
                    input_shape=(3,),
                    kernel_initializer=
                    tfk.initializers.RandomNormal(mean=0, stddev=5)
                    )
              )

    for j in range(depth - 1):
        model.add(Dense(width,
                        activation="sigmoid")
                  )

    model.add(Dense(3))

    model.compile(loss="mse",
                  optimizer=tfk.optimizers.RMSprop(learning_rate=0.01),
                  metrics=["accuracy"]
                  )

    return model

def train_model(model, X_train, Y_train, X_test, Y_test, epochs = 10):
    history = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test, Y_test)
                        )

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'r', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    return history


if __name__ == "__main__":

    # setup
    np.random.seed(2002)
    dt = 0.01; T = 8; t = np.arange(0, T, dt)
    net = compile_model()

    n_states = 1000
    train_split = 0.75

    # uniformly distributed points
    states = 30*np.random.rand(n_states, 3)

    # create next states
    next_states = np.zeros((n_states, 3))
    for j in range(n_states):
        next_states[j] = Lorenz(dt, states[j])

    # split into train and test
    indices = np.random.permutation(states.shape[0])

    train_ind = indices[:int(n_states*train_split)]
    test_ind = indices[int(n_states * train_split):]

    X_train = states[train_ind]
    Y_train = next_states[train_ind]

    X_test = states[test_ind]
    Y_test = next_states[test_ind]

    # train net
    train_model(net, X_train, Y_train, X_test, Y_test)

    # get a particular trajectory, and follow it
    x_start = np.array([-5, -15, 10])
    end_iter = int(T/dt)
    flow = np.zeros((end_iter + 1, 3))
    flow[0] = x_start

    for i in range(end_iter):
        flow[i+1] = Lorenz(dt, flow[i])








