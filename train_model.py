from neural_network_model import *
from text_conversions import *
import pickle as pkl
import numpy as np
import json
import os


def TrainModel(intents):
    train_x = []  # train x are the inputs for training
    train_y = []  # train y are the correct outputs for training
    vocabulary = []  # vocabulary is all the words needed for converting to bag of words format
    labels = []  # labels are all the labels that our text can fall under

    model_name = intents['intents']['name']

    # adds data from the json file into our variables
    for intent in intents['intents']['data']:
        labels.append(intent['label'])
        for sentence in intent['patterns']:
            base_words = SentenceToBaseWords(sentence)
            vocabulary.extend(base_words)
            train_x.append(base_words)
            train_y.append(labels.index(intent['label']))

    # converts all words to the bag of words format
    train_x = [WordsToBOW(words, vocabulary) for words in train_x]
    # converts all labels to one-hot-encoding for our neural network to understand
    train_y = [LabelsToOneHot(class_id, len(labels)) for class_id in train_y]

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # shuffles the training data uniformly to prevent bias
    shuffler = np.random.permutation(len(train_x))
    train_x = train_x[shuffler]
    train_y = train_y[shuffler]

    # create the layers
    dense1 = Dense(train_x.shape[1], 64)
    activation1 = Relu()
    dense2 = Dense(64, train_y.shape[1])
    activation2 = Softmax()
    loss_activation = SoftmaxWithCategoricalCrossEntropy()
    optimizer = SGD(decay=1e-3, momentum=0.2)

    # training (I have not included batches since the dataset is small and the model is still accurate when I test it)
    # I have tested the model myself instead of using test data
    for epoch in range(1001):
        layer_output = dense1.forward(train_x)
        layer_output = activation1.forward(layer_output)
        layer_output = dense2.forward(layer_output)
        loss = np.mean(loss_activation.forward(layer_output, train_y))
        predictions = np.argmax(loss_activation.softmax_outputs, axis=1)
        accuracy = np.mean(predictions == np.argmax(train_y, axis=1))
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy}, ' +
                  f'loss: {loss}')

        loss_activation.backward()
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_decay()
        optimizer.update_layer(dense1)
        optimizer.update_layer(dense2)

    # saves the layers so that we can use the model later
    folder = f'data/{model_name}'
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(f'{folder}/layer1', 'wb') as file:
        pkl.dump(dense1, file)

    with open(f'{folder}/activation1', 'wb') as file:
        pkl.dump(activation1, file)
    with open(f'{folder}/layer2', 'wb') as file:
        pkl.dump(dense2, file)
    with open(f'{folder}/activation2', 'wb') as file:
        pkl.dump(activation2, file)
    with open(f'{folder}/vocabulary', 'wb') as file:
        pkl.dump(vocabulary, file)
    with open(f'{folder}/labels', 'wb') as file:
        pkl.dump(labels, file)

    return loss, accuracy
