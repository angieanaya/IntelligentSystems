'''
@File    :   KNN_Scratch.py
@Date    :   2022/04/06
@Author  :   María de los Ángeles Contreras Anaya
@Version :   1.0
@Desc    :   Program that implements the K Nearest Neighbors algorithm from scratch to predict breast cancer.
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def split(X, y, train_size):
    """
    Splits features and labels into training and testing sets

    Parameters:
        X: DataFrame that contains all features from the dataset
        y: Series of all the values from the label column of the dataset
        train_size: Represents the proportion of the dataset to include in the train split.

    Returns:
        X_train (Array): Training set of the feature values
        X_test (Array): Test set of the feature values
        y_train (Array): Training set of the label values
        y_test (Array): Test set of the label values
    """
    shuffle_x = X.sample(frac=1, random_state=2)
    shuffle_y = y.sample(frac=1, random_state=2)

    train_size = int(train_size * len(dataset))

    X_train = shuffle_x[:train_size].values
    X_test = shuffle_x[train_size:].values
    y_train = shuffle_y[:train_size].values
    y_test = shuffle_y[train_size:].values
    
    return X_train, y_train, X_test, y_test

def get_distance(p1,p2):
    """
    Calculates the distance between two given points using the Euclidean measure

    Parameters:
        p1: Array of elements from the features training set
        p2: Array of elements from the features testing set

    Returns:
        The distance between the given points
    """
    return np.sqrt(np.sum((p1-p2)**2))

def get_mode(values):
    """
    Gets the mode from a given list, if two modes are found it returns the smallest one

    Parameters:
        values: List of labels from the K-nearest neighbors

    Returns:
        modes(int): The first value of the list of modes, which is the smallest one.
    """
    modes = []
    counts = {k:values.count(k) for k in set(values)}
    for key, val in counts.items():
        if val == max(counts.values()):
            modes.append(key)
    modes.sort()
    return modes[0]

def get_neighbors(X_train, test_row, k): 
    """
    Predicts breast cancer based on given input using the K-Nearest algorithm

    Parameters:
        X_train: Training set that contains the series of feature values
        test_row: Row of features from the testing set.
        k: Number of neighbors required for each sample.

    Returns:
        k_nearest_neighbors (Array): All predictions made for the given feature values.
    """
    neighbors = []

    for i in range(len(X_train)): 
        distances = get_distance(np.array(X_train[i,:]), test_row) 
        neighbors.append(distances) 
    neighbors = np.array(neighbors) 
         
    k_nearest_neighbors = np.argsort(neighbors)[:k] 
    return k_nearest_neighbors

def predict(X_train, y_train , X_test, k):
    """
    Predicts breast cancer based on given input using the K-Nearest algorithm

    Parameters:
        X_train: Set that contains the series of feature values for training purposes.
        y_train: Set that contains the series of label values for training purposes.
        X_test: Set that contains the feature values for testing purposes.
        k: Number of neighbors required for each sample.

    Returns:
        classification (Array): All predictions made for the given feature values.
    """
    classification = []

    for elem in X_test: 
        k_nearest_neighbors = get_neighbors(X_train, elem, k)
        
        labels = y_train[k_nearest_neighbors]
        classification.append(get_mode(labels.tolist()))
 
    return classification

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

def get_k(X_train, y_train, X_test, y_test):
    """
    Calculates the optimal K values for the algorithm

    Parameters:
        X_train: Set that contains the series of feature values for training purposes.
        y_train: Set that contains the series of label values for training purposes.
        X_test: Set that contains the feature values for testing purposes.
        y_test: Set that contains the label values for testing purposes.

    Returns:
       optimal_k (int): The best value for K, meaning the value with the least error rate.
    """
    k_error = {}
    error = []
    for i in range(1, 25):
        pred_i = predict(X_train,y_train,X_test, i)
        error.append(np.mean(pred_i != y_test))
        k_error[i] = np.mean(pred_i != y_test)
    optimal_k = min(k_error, key=k_error.get)
    
    # Plotting the error for k number of nearest neighbors
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 25), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate - K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.savefig('IntelligentSystems/GenerativeVSDiscriminative/results/optimal_k.png')
    return optimal_k

# Loading the dataset
dataset = pd.read_csv("IntelligentSystems/GenerativeVSDiscriminative/data.csv")

# Preprocessing dataset
dataset['diagnosis'] = encoder(dataset['diagnosis']) 
dataset = dataset.iloc[: , :-1]
dataset = dataset.iloc[: , 1:]

# Dividing dataset into features and labels
feature_columns = ['concave points_worst', 'perimeter_worst', 'concave points_mean']
X = dataset[feature_columns]

# USING THE WHOLE DATASET
#X = dataset.loc[:, dataset.columns != "diagnosis"]

y = dataset['diagnosis']

# Splitting data
X_train, y_train, X_test, y_test = split(X, y, 0.8)

# TO GET THE OPTIMAL VALUE OF K
#optimal_k = get_k(X_train, y_train, X_test, y_test)

optimal_k = 4 # SINCE I KNOW THE OPTIMAL VALUE FOR K IS 4

#Applying the KNN algorithm from scratch 
y_pred = predict(X_train,y_train,X_test, optimal_k)

# Plot confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap='GnBu')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.xaxis.set_ticklabels(['Malign', 'Benign']) 
ax.yaxis.set_ticklabels(['Malign', 'Benign'])
plt.savefig('IntelligentSystems/GenerativeVSDiscriminative/results/KNN_confusion_matrix_.png')

# Get and display metrics from model
accuracy = "{:.2f}".format(get_accuracy(y_test, y_pred))
precision = "{:.2f}".format(get_precision(confusion_matrix))
recall = "{:.2f}".format(get_recall(confusion_matrix))
metrics = PrettyTable()
metrics.padding_width = 6
metrics.field_names = ["Accuracy", "Precision", "Recall"]
metrics.add_row([accuracy+" %", precision+" %", recall+" %"])
title = f"Metrics for the K-nearest neighbor algorithm with K = {optimal_k}"
table = metrics.get_string(title=title)
print(table)
with open('IntelligentSystems/GenerativeVSDiscriminative/results/metrics_KNN.txt', 'w') as f:
    f.write(table)