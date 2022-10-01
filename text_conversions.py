import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def SentenceToBaseWords(sentence, ignore=['.', '!', ',', '?', 'the', 'an', 'a']):
    # gets a list of all the words in a sentence
    words = word_tokenize(sentence)
    # removes words and/or punctuation that need to be ignored
    words = filter(lambda w: w not in ignore, words)
    words = [WordNetLemmatizer().lemmatize(w.lower())
             for w in words]  # converts all words to their base form
    return words


def WordsToBOW(words, vocabulary):
    # converts each sentence to bag of words format
    return list(map(lambda w: words.count(w), vocabulary))


def LabelsToOneHot(label, max_label):
    one_hot = [0 for i in range(max_label)]
    one_hot[label] = 1
    return one_hot


def OneHotToLabels(one_hot, labels):
    return labels[one_hot.index(max(one_hot))]
