# Text-Classification-API

This is a text classification API that aLlows users to create models that can predict text from a database, and then use these to classify a given sentence. It uses Flask to create the API because Flask has allowed me to make the APIs quickly without any unnecessary boilerplate. I have used Numpy to help me create the neural network because it offers faster runtime and reduces development time.

## Limitations of the Model
I have used the 'Bag of Words' technique to classify the text. Although this allows users to train models with very small datasets, models struggle to understand the context of sentences or paragraphs because it does not take into account the order of words. Because of this, they are best suited for analysing small texts such as different types of sentences that an AI chatbot might need to recognise in order to respond to.

## Using the Model
In order to start using the API, it will need to be hosted by running the flask_api.py file. The code currently hosts on localhost, but this can be changed in the file.

After the API is running, you can use the route '/create_model' to create a model from a dataset. This is a POST request and data will need to be posted in the format of the example_intentes.json file. The accuracy and loss of the model are shown on the web page.

Once your model is created you can start to use it. This can be done in 2 ways. The first way is using the GET request with the route, '/predict/{model name}/{sentence}' where the model name and sentence (to predict) are parameters. This will return a JSON response with the format: {'sentence': sentence, 'prediction': prediction}.

An alternative method of predicting text is using a POST request. In this, you use the route, '/predict/_/_' and post the data in the format: {'name': model_name, 'sentence': sentence}. The response will be in the same format described above.
