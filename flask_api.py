from flask import Flask
from flask import request
from train_model import *
from model_predict import *

app = Flask(__name__)


@app.route('/')
def home():
    return '''Welcome to the chatbot api.</br>
    Use "/create_model" to create a new chatbot by POSTing the intents.</br>
    Use "/predict/[model_name]/[sentence]" to GET the label for a given sentence.'''


@app.route('/predict/<model_name>/<sentence>', methods=['GET'])
def predict_text(model_name, sentence):
    response = {'prediction': ModelPredict(model_name, sentence)}
    return response


@app.route('/create_model', methods=['POST'])
def create_model():
    loss, accuracy = TrainModel(request.get_json())
    return f'loss: {loss} </br> accuracy: {accuracy}'


if __name__ == '__main__':
    app.run()
