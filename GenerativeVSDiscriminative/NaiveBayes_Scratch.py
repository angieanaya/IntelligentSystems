'''
@File    :   NaiveBayes_Scratch.py
@Date    :   2022/04/07
@Author  :   María de los Ángeles Contreras Anaya
@Version :   1.0
@Desc    :   Program that implements the Naive Bayes algorithm with a normal distribution from scratch to predict breast cancer.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

def encoder(column):
    """
    Converts categorical values from the given column into numbers as part of preprocessing the data.

    Parameters:
        column: Column with a series of n categorical values from dataset

    Returns:
        column (DataFrame): Transformed column
    """
    values = column.values.tolist()
    categories = []
    for value in values:
        if(value not in categories):
            categories.append(value)
    for value in values:
        for category in categories:
            if(value == category):
                index = values.index(value)
                numerical_value = categories.index(category)
                values = values[:index]+[numerical_value]+values[index+1:]
    column = pd.DataFrame(values)   
    return column

def split(dataset, train_size):
    """
    Splits dataset into training and testing sets

    Parameters:
        dataset: The loaded dataset.
        train_size: Represents the proportion of the dataset to include in the train split.

    Returns:
        train_set (DataFrame): Part of the dataset that is destined for training the algorithm
        test_set (DataFrame): Part of the dataset that is destined for testing the algorithm
    """
    shuffle_dataset = dataset.sample(frac=1, random_state=2)

    train_size = int(train_size * len(dataset))

    train_set = shuffle_dataset[:train_size]
    test_set = shuffle_dataset[train_size:]
    
    return train_set, test_set

def get_prior_probability(train_set, y):
    """
    Calculates the probability of the data being classified as 1 or 0 based on current knowledge.

    Parameters:
        train_set: Training split of the dataset
        y: Name of the label 

    Returns:
        probability (list): List of the probabilities of the event being classified as 0 or 1
    """
    categories = sorted(train_set[y].unique())
    probability = []
    for i in categories:
        probability.append(len(train_set[train_set[y]==i])/len(train_set))
    return probability

def get_stats(train_set, feature):
    """
    Calculates the mean and standard deviation of the given features

    Parameters:
        train_set: Training split of the dataset
        feature: Feature from the dataset on which to perform the functions

    Returns:
        mean (float): Mean of the given feature
        std (float): Standard deviation of the given feature
    """
    mean = train_set[feature].mean()
    std = train_set[feature].std()
    return mean, std

def get_gaussian_distribution(train_set, feature, feature_value, y, category):
    """
    Calculates the probability with the Gaussian (normal) distribution of an event ocurring.

    Parameters:
        train_set: Training split of the dataset
        feature: Feature from the dataset on which to perform the functions
        feature_value: Value of the given feature
        y: Name of the label column
        category: The classification of that event ocurring

    Returns:
        probability (float): Gaussian distribution
    """
    train_set = train_set[train_set[y]==category]
    mean,std = get_stats(train_set, feature)
    probability = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feature_value-mean)**2 / (2 * std**2)))
    return probability

def predict(train_set, X_test, y):
    """
    Calculates the probability with the Gaussian (normal) distribution of an event ocurring.

    Parameters:
        train_set: Training split of the dataset
        X_test: Testing split containing only the features.
        y: Name of the label column

    Returns:
        y_pred (list): List of all predictions for the given inputs.
    """
    features = list(train_set.columns)[1:]

    prior_prob = get_prior_probability(train_set, y) # fitting the model

    # predicting
    y_pred = []
    for x in X_test:
        categories = sorted(train_set[y].unique())
        likelihood = [1]*len(categories)
        
        for j in range(len(categories)):
            for i in range(len(features)):
                likelihood[j] *= get_gaussian_distribution(train_set, features[i], x[i], y, categories[j])

        posterior_prob = [1]*len(categories)
        
        for j in range(len(categories)):
            posterior_prob[j] = likelihood[j] * prior_prob[j]

        y_pred.append(np.argmax(posterior_prob))
    return y_pred

def get_accuracy(y_test, y_pred):
    """
    Computes the accuracy between the actual values and the performed predictions

    Parameters:
        y_test: Array holding the set of actual values for the labels
        y_pred: List of redicted values with the KNN algorithm

    Returns:
        accuracy_metric (float): Accuracy metric for the model
    """
    correct_predictions = 0
    total_predictions = len(y_test)
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct_predictions += 1
    accuracy_metric = correct_predictions / total_predictions * 100
    return accuracy_metric

def confusion_matrix(y_test, y_pred):
    """
    Returns the confusion matrix for the given actual values and predictions
    
    Parameters:
        y_test: Array holding the set of actual values for the labels
        y_pred: List of predicted values with the KNN algorithm 

    Returns:
        confusion_matrix (list): Two dimensional list with the actual and predicted values for confusion matrix construction
    """
    list_options = set(y_test)
    num_predictions = len(y_test)
    
    confusion_matrix = [list(0 for option in (list_options)) for option in (list_options)]
    
    values = {}
    for i, value in enumerate(list_options):
        values[value] = i
    
    for i in range(num_predictions):
        x = values[y_test[i]]
        y = values[y_pred[i]]
        confusion_matrix[x][y] += 1
        
    return confusion_matrix

def get_precision(matrix):
    """
    Computes the precision of the performed predictions based on the actual values.

    Parameters:
        matrix: Confusion Matrix of the model to get the TP and FP values

    Returns:
        precision_metric (float): Precision metric for the model
    """
    tp = matrix[0][0] 
    fp = matrix[1][0]
    precision_metric =tp/ (tp + fp) * 100
    return precision_metric

def get_recall(matrix):
    """
    Computes the recall metric for the implemented model 

    Parameters:
        matrix: Confusion Matrix of the model to get the TP and FN values

    Returns:
        recall_metric (float): Recall metric for the model
    """
    tp = matrix[0][0] 
    fn = matrix[0][1]
    recall_metric = tp/ (tp + fn) * 100
    return recall_metric

# Loading the dataset
dataset = pd.read_csv("IntelligentSystems/GenerativeVSDiscriminative/data.csv")

# Preprocessing dataset
label = "diagnosis"
dataset[label] = encoder(dataset[label]) 
dataset = dataset.iloc[: , :-1]
dataset = dataset.iloc[: , 1:]
dataset = dataset.loc[:, ['diagnosis','concave points_worst', 'perimeter_worst', 'concave points_mean']]

# Splitting data
train, test = split(dataset, 0.8)
X_test = test.loc[:, test.columns != label].values
y_test = test[label].values

y_pred = predict(train, X_test, label)

# Get and display metrics from model
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = "{:.2f}".format(get_accuracy(y_test, y_pred))
precision = "{:.2f}".format(get_precision(confusion_matrix))
recall = "{:.2f}".format(get_recall(confusion_matrix))
metrics = PrettyTable()
metrics.padding_width = 6
metrics.field_names = ["Accuracy", "Precision", "Recall"]
metrics.add_row([accuracy+" %", precision+" %", recall+" %"])
title = f"Metrics for the Naive Bayes algorithm with the Gaussian Distribution"
table = metrics.get_string(title=title)
print(table)
with open('IntelligentSystems/GenerativeVSDiscriminative/results/metrics_NaiveBayes.txt', 'w') as f:
    f.write(table)

# Plot confusion matrix
ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap='GnBu')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.xaxis.set_ticklabels(['Malign', 'Benign']) 
ax.yaxis.set_ticklabels(['Malign', 'Benign'])
plt.savefig('IntelligentSystems/GenerativeVSDiscriminative/results/NB_confusion_matrix.png')