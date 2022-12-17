from math import factorial, sin, cos, tan

steps_on_range = 2000
batch_size_of_sequence = 20
length_of_sequence = 100
number_of_epochs = 20
start_value = -50
end_value = 50
activation_function = "tanh"
train_size_of_sequence = 0.9
function_of_sequences = {
    " 1/(2^x)".strip(): lambda x: 1 / (2 ** x),
    " x".strip(): lambda x: x,
    " (-1)^(x-1)*x".strip(): lambda x: (-1) ** ((x - 1) * x),
    " x!".strip(): lambda x: factorial(x),
    " sin(x)".strip(): lambda x: sin(x),
    " cos(x)".strip(): lambda x: cos(x),
    " tan(x)".strip(): lambda x: tan(x)
}
