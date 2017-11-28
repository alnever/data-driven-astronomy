import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

import itertools


def splitdata_train_test(data, fraction_training):
  # complete this function
  np.random.seed(0)
  np.random.shuffle(data)
  split = math.trunc(data.shape[0] * fraction_training)
  training = data[:split]
  testing = data[split:]
  return training, testing  

def generate_features_targets(data):
  # complete the function by calculating the concentrations

  targets = data['class']

  features = np.empty(shape=(len(data), 13))
  features[:, 0] = data['u-g']
  features[:, 1] = data['g-r']
  features[:, 2] = data['r-i']
  features[:, 3] = data['i-z']
  features[:, 4] = data['ecc']
  features[:, 5] = data['m4_u']
  features[:, 6] = data['m4_g']
  features[:, 7] = data['m4_r']
  features[:, 8] = data['m4_i']
  features[:, 9] = data['m4_z']

  # fill the remaining 3 columns with concentrations in the u, r and z filters
  # concentration in u filter
  features[:, 10] = data['petroR50_u'] / data['petroR90_u']
  # concentration in r filter
  features[:, 11] = data['petroR50_r'] / data['petroR90_r']
  # concentration in z filter
  features[:, 12] = data['petroR50_z'] / data['petroR90_z']

  return features, targets

# complete this function by splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
  # split the data into training and testing sets using a training fraction of 0.7
  fraction_training = 0.7
  training, testing = splitdata_train_test(data, fraction_training)
  # generate the feature and targets for the training and test sets
  # i.e. train_features, train_targets, test_features, test_targets
  training_features, training_targets = generate_features_targets(training)
  testing_features, testing_targets = generate_features_targets(testing)

  # instantiate a decision tree classifier
  dtr = DecisionTreeClassifier()

  # train the classifier with the train_features and train_targets
  dtr.fit(training_features, training_targets)

  # get predictions for the test_features
  predictions = dtr.predict(testing_features)

  # return the predictions and the test_targets
  return predictions, testing_targets

# Implement the following function
def calculate_accuracy(predicted, actual):
  return sum((predicted == actual)) / len(predicted)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
  

if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # set the fraction of data which should be in the training set
  fraction_training = 0.7

  # split the data using your function
  training, testing = splitdata_train_test(data, fraction_training)

  # print the key values
  print('Number data galaxies:', len(data))
  print('Train fraction:', fraction_training)
  print('Number of galaxies in training set:', len(training))
  print('Number of galaxies in testing set:', len(testing))

  features, targets = generate_features_targets(data)

  # Print the shape of each array to check the arrays are the correct dimensions. 
  print("Features shape:", features.shape)
  print("Targets shape:", targets.shape)

  # Predictions 
  predicted_class, actual_class = dtc_predict_actual(data)

  # Print some of the initial results
  print("Some initial results...\n   predicted,  actual")
  for i in range(10):
    print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))


  # calculate the model score using your function
  model_score = calculate_accuracy(predicted_class, actual_class)
  print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(actual_class))
  model_cm = confusion_matrix(y_true=actual_class, y_pred=predicted_class, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()