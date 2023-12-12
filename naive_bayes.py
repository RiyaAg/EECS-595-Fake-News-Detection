import os
from os import listdir
import re
import nltk
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import GridSearchCV

class Classifier(object):

    def __init__(self):
        # setup, define classes folder names and initialize vectorizer and classifier
        self.count_vectorizer = CountVectorizer(ngram_range=(1,5))
        self.classifier = MultinomialNB()

        pass

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    def preprocess(self, data):


        # set up word preprocessing
        processed = []
        lemmatizer = WordNetLemmatizer()
        for review in data: 

            # remove special chars and numbers, set to all lowercase
            review = re.sub(r'[^\w\s\d]', '', review)

            #tokenize, remove stopwords, lemmatize
            tokens = word_tokenize(review)

            stopword_dict = stopwords.words('english')
            tokens = [word for word in tokens if word not in stopword_dict]

            tokens = [lemmatizer.lemmatize(word) for word in tokens]


            #join the tokens back together
            review = ' '.join(tokens)

            #add processed entry to dictionary
            processed.append(review)
        print("processed data", len(processed), len(processed[1]))

        return processed

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    def predict(self, test_data, test_labels):

        # load and preprocess testing data and set ground truth to labels 
        #   testing_data, ground_truth = self.load(testing_path)
        ground_truth = test_labels
        testing_data = self.preprocess(test_data)
        # bag of words feature extraction
        test = self.count_vectorizer.transform(testing_data)
        model_predictions = self.classifier.predict(test)

        return model_predictions, ground_truth

     # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #   



def train(train_data, train_labels):
    """
        Trains the naive Bayes classifier.
        Implement data loading, pre-processing, and model training here

        Params:
        training_path (string):
        string for the file location of the training data (the "training" directory).
        Return:
        output_string (string):
        An object representing the trained model.

    # """
    classifier = Classifier()
    preprocessed_training = classifier.preprocess(train_data)
    param_grid = {'alpha': np.linspace(0.1, 2.0, 10)}  # Example alpha values
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    X_train = classifier.count_vectorizer.fit_transform(preprocessed_training)
    
    grid_search.fit(X_train, train_labels)
    best_alpha = grid_search.best_params_['alpha']

    # Update the classifier with the best hyperparameter
    classifier.classifier = MultinomialNB(alpha=best_alpha)
    print(best_alpha)
    classifier.classifier.fit(X_train, train_labels)

    return classifier



def predict(trained_model, test_data, test_label):
    """
        Runs prediction of the trained naive Bayes classifier on the test set, and returns these predictions.
        Implement data loading, preprocessing, and model prediction

        Params:
        trained_model (object):
        An object representing the trained model (whatever is returned by the above function)
        testing_path (string):
        A string for the file location of the test data (the "testing" directory)
        Return:
        model_predictions:
        An object representing the predictions of the trained model on the testing data
        ground_truth:
        An object representing the ground truth labels of the testing data.

    """
    model_predictions = []
    ground_truth = []


    # after training the model, use it to predict the testing data
    model_predictions, ground_truth = trained_model.predict(test_data, test_label)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model_predictions, ground_truth


def evaluate(model_predictions, ground_truth):
    """
    Evaluates the accuracy of model predictions using the ground truth labels.
    Implement evaluation metrics for the predictions

    Params:
    model_predictions:
    An object representing the predictions of the trained model on the testing data
    ground_truth:
    An object representing the ground truth labels of the testing data.
    Return:
    accuracy (float):
    model_predictions:
    Floating-point accuracy of the trained model on the test set.

    """

    #use the ground truths to check the accuracy
    accuracy = accuracy_score(ground_truth, model_predictions)

    return accuracy
#     = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
