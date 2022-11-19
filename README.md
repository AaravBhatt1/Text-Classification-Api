# Text-Classification-API

This is a text classification API that aLlows users to create models that can predict text from a database, and then use these to classify a given sentence. It uses Flask to create the API because Flask has allowed me to make the APIs quickly without any unnecessary boilerplate. I have used Numpy to help me create the neural network because it offers faster runtime and reduces development time.

## Limitations of the Model
I have used the 'Bag of Words' technique to classify the text. Although this allows users to train models with very small datasets, models struggle to understand the context of sentences or paragraphs because it does not take into account the order of words. Because of this, they are best suited for analysing small texts such as different types of sentences that an AI chatbot might need to recognise in order to respond to. 
