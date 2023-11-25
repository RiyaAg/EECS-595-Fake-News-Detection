    # For Problem 5,
    #
    # We will implement a naive Bayesian sentiment classifier which learns to classify movie reviews as positive or negative.
    #
    # Please implement the following functions below: train(), predict(), evaluate().
    # Feel free to use any Python libraries such as sklearn, numpy, etc.
    # DO NOT modify any function definitions or return types, as we will use these to grade your work.
    # However, feel free to add new functions to the file to avoid redundant code (e.g., for preprocessing data).
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
        # self.classes = ["pos", "neg"]
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
    # trained_model = Classifier()
    # # initialize classifier class
    # #load and preprocess training data
    # # training_data, training_labels = trained_model.load(training_path)
    # print("loaded training data")
    # # print("length:", len(training_data))


    # preprocessed_training = trained_model.preprocess(train_data)

    # #bag of words unigram extraction and fir classifier to training data
    # train = trained_model.count_vectorizer.fit_transform(preprocessed_training)
    # trained_model.classifier.fit(train, train_labels)

    # # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    # return trained_model
        # Initialize the classifier
    classifier = Classifier()

    # Preprocess training data
    preprocessed_training = classifier.preprocess(train_data)

    # Perform grid search for hyperparameter tuning
    param_grid = {'alpha': np.linspace(0.1, 2.0, 10)}  # Example alpha values
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    
    # Fit the vectorizer on the training data
    X_train = classifier.count_vectorizer.fit_transform(preprocessed_training)
    
    grid_search.fit(X_train, train_labels)

    # Get the best hyperparameter value
    best_alpha = grid_search.best_params_['alpha']

    # Update the classifier with the best hyperparameter
    classifier.classifier = MultinomialNB(alpha=best_alpha)

    # Fit the classifier to the training data
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

# #  return accuracy
# import re
# import nltk
# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk import word_tokenize

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# class Classifier(object):

#     def __init__(self):
#         # setup, define classes folder names and initialize vectorizer and classifier
#         self.count_vectorizer = CountVectorizer(ngram_range=(1, 5))
#         self.classifier = MultinomialNB()

#     def preprocess(self, data):
#         processed = []
#         lemmatizer = WordNetLemmatizer()

#         for review in data:
#             review = re.sub(r'[^\w\s\d]', '', review.lower())  # Combine lowercase and special character removal
#             tokens = word_tokenize(review)
#             stopword_dict = set(stopwords.words('english'))
#             tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword_dict]
#             review = ' '.join(tokens)
#             processed.append(review)

#         return processed

#     def predict(self, test_data):
#         testing_data = self.preprocess(test_data)
#         test = self.count_vectorizer.transform(testing_data)
#         model_predictions = self.classifier.predict(test)
#         return model_predictions

# def tune_and_train(train_data, train_labels):
#     # Initialize the classifier
#     classifier = Classifier()

#     # Preprocess training data
#     preprocessed_training = classifier.preprocess(train_data)

#     # Perform grid search for hyperparameter tuning
#     param_grid = {'alpha': np.linspace(0.0000001, 0.00001, 1)}  # Example alpha values
#     grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    
#     # Fit the vectorizer on the training data
#     X_train = classifier.count_vectorizer.fit_transform(preprocessed_training)
    
#     grid_search.fit(X_train, train_labels)

#     # Get the best hyperparameter value
#     best_alpha = grid_search.best_params_['alpha']
#     print(best_alpha)
#     # Update the classifier with the best hyperparameter
#     classifier.classifier = MultinomialNB(alpha=best_alpha)

#     # Fit the classifier to the training data
#     classifier.classifier.fit(X_train, train_labels)

#     return classifier

# def evaluate( model_predictions, test_label):   # Preprocess testing data
#     # testing_data = trained_model.preprocess(test_data)

#     # # Transform the testing data using the fitted vectorizer
#     # X_test = trained_model.count_vectorizer.transform(testing_data)

#     # # Predict using the trained model
#     # model_predictions = trained_model.predict(X_test)

#     # Evaluate accuracy
#     accuracy = accuracy_score(test_label, model_predictions)

#     return accuracy