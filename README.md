# AI-Chatbot-from-Scratch

This is an AI chatbot that classifies texts into different labels.
It can be trained from a very small dataset that can be given through a POST API. 
It can then predict the meaning of a sentence and then return this back through a GET API.

The neural network consists of 1 hidden layer with 64 neurons. 
This is then connected to a RELU activation function, which creates non-linearity and then connects to the output layer.
The output layer then uses a softmax acttivation function that gives probabilites for each label based on whether the input sentence has that meaning.
I have used the SGD optimizer simply because it is the least complex to code, but it still performs accurately in tests because of the simplicity of the task.
The book Neural Networks from Scratch has helped me fill any gaps in my knowledge, and it particularly helped explain the optimiser and weight regularisation to me.

Because this project does not revolve around complex API responses or pretty GUIs, I used the Flask framework to create my APIs because it is much simpler than other options.
It also minimised the development time for this project.

The example_intents.json file shows the format in which the json data for creating a model should look like in the POST API request

If I were to improve this chatbot in the future, I could:
- maybe change the optimizer to the ADAM optimizer and perhaps use batches to train the neural network if the dataset is large.
- create a GUI for creating and using the chatbot.
