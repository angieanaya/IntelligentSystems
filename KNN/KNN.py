import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from pylab import rcParams


def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 25
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".1f")
    fig.savefig('correlation_coefficient.png')

# Loading the dataset
dataset = pd.read_csv("PCOS_data.csv")

# plot correlation & densities
plot_correlation(dataset)

# Dividing dataset into features and labels
feature_columns = ['Follicle No. (R)', 'Follicle No. (L)', 'hair growth(Y/N)', 'Skin darkening (Y/N)']
worst_feature_columns = ['FSH/LH', 'Cycle length(days)', 'PRG(ng/mL)', 'RBS(mg/dl)']
noncorrelated_feature_columns = ['Follicle No. (R)', 'hair growth(Y/N)', 'Skin darkening (Y/N)']

X = dataset[feature_columns].values
y = dataset['PCOS (Y/N)'].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Finding the best K value using an error plot
k_error = {}
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    k_error[i] = np.mean(pred_i != y_test)

optimal_k = min(k_error, key=k_error.get)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate - K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig('optimal_k.png')

# Applying KNN algorithm
classifier = KNeighborsClassifier(n_neighbors=optimal_k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluating performance
accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred)*100)
print("********************************************************************************")
print(f"The model has an accuracy of {accuracy}% using K={optimal_k} nearest neighbors.")
print("********************************************************************************")
plt.figure(figsize=(12, 6))
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['PCOS','No PCOS'])
cmd.plot(cmap='GnBu')
cmd.ax_.set_title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.clf()

target = ('PCOS', 'No PCOS')
clf_report=classification_report(y_test, y_pred, target_names=target, output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="BuPu")
plt.title("Classification Report")
plt.savefig('classification_report.png')