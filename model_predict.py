from text_conversions import *
import numpy as np
import os
import pickle as pkl


def ModelPredict(model_name, sentence):
    # load the neural network and other important data from the model folder
    folder = f'data/{model_name}'
    if not os.path.exists(folder):
        return
    with open(f'{folder}/layer1', 'rb') as file:
        dense1 = pkl.load(file)
    with open(f'{folder}/activation1', 'rb') as file:
        activation1 = pkl.load(file)
    with open(f'{folder}/layer2', 'rb') as file:
        dense2 = pkl.load(file)
    with open(f'{folder}/activation2', 'rb') as file:
        activation2 = pkl.load(file)
    with open(f'{folder}/vocabulary', 'rb') as file:
        vocabulary = pkl.load(file)
    with open(f'{folder}/labels', 'rb') as file:
        labels = pkl.load(file)

    # return the output
    words = SentenceToBaseWords(sentence)
    words = WordsToBOW(words, vocabulary)
    words = np.array([words])
    output = dense1.forward(words)
    output = activation1.forward(output)
    output = dense2.forward(output)
    output = activation2.forward(output)
    output = output.tolist()[0]
    return OneHotToLabels(output, labels)
