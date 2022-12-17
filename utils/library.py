from tqdm import tqdm
from utils.config import length_of_sequence, function_of_sequences


def get_input_function():
    while True:
        sequence = input("Input function: ").strip().lower()
        if function_of_sequences.get(sequence) is None:
            print("This function doesn't exist")
        else:
            func = function_of_sequences.get(sequence)
            break
    return func


def training(LSTM_model, train, number_of_epochs) -> None:
    for i in range(number_of_epochs):
        LSTM_model.fit(train, validation_data=None)


def testing_reuslts(LSTM_model, test_seq) -> list:
    result = [test_seq[i][0] for i in range(length_of_sequence)]
    begin = [test_seq[:length_of_sequence]]
    length = len(test_seq) - length_of_sequence
    for i in tqdm(range(length)):
        next_step = float(LSTM_model.predict(begin, verbose=0)[0][0])
        begin[0] = begin[0][1:]
        begin[0].append([next_step])
        result.append(next_step)

    return result
