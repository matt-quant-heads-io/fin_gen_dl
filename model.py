from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, LSTM


def compile_6_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    # Layer 2
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 3
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 4
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 5
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 6
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model

def compile_5_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    # Layer 2
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 3
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 4
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 5
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model

def compile_4_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    # Layer 2
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 3
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 4
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model

def compile_3_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    # Layer 2
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    # Layer 3
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model

def compile_2_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    # Layer 2
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, 100)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model

def compile_1_layer_model(loss='mse', num_units=None, activation_func='relu', dropout=0.1, alpha=0.5,
                          n_inputs=None, n_features=None, output_units=None, optimizer='adam'):
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=num_units,
                   activation=activation_func,
                   input_shape=(n_inputs, n_features)))

    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_units))

    model.compile(optimizer, loss)

    return model


