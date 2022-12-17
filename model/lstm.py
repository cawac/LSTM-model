import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.config import train_size_of_sequence, batch_size_of_sequence, activation_function, length_of_sequence, start_value, end_value, steps_on_range, number_of_epochs
from utils.library import training, testing_reuslts, get_input_function


def start_system():
    function_for_aproximation = get_input_function()
    range_of_dataset = np.linspace(start_value, end_value, steps_on_range)
    data = [[function_for_aproximation(i)] for i in range_of_dataset]

    train_sequence_set = data[:int(train_size_of_sequence * len(data))]
    test_sequence_set = data[int(train_size_of_sequence * len(data)):]
    train_target_set = train_sequence_set[length_of_sequence:]
    test_range_of_data_set = range_of_dataset[int(train_size_of_sequence * len(data)):]

    train_sequence = tf.keras.utils.timeseries_dataset_from_array(sequence_length=length_of_sequence, targets=train_target_set, batch_size=batch_size_of_sequence, data=train_sequence_set,)

    LSTMModel = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(length_of_sequence, 1)), tf.keras.layers.LSTM(number_of_epochs, activation=activation_function, return_sequences=False), tf.keras.layers.Dense(units=1)])

    LSTMModel.compile(loss=tf.losses.Huber(), optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=[tf.metrics.MeanAbsoluteError()])

    LSTMModel.build()
    LSTMModel.summary()

    training(LSTMModel, train_sequence, number_of_epochs)
    forecast_diagram = testing_reuslts(LSTMModel, test_sequence_set)

    plt.figure(figsize=(12, 7))
    plt.plot(test_range_of_data_set, test_sequence_set)
    plt.plot(test_range_of_data_set, forecast_diagram)
    plt.show()


if __name__ == '__main__':
    start_system()
