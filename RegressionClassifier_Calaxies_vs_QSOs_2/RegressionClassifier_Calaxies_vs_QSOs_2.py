import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

# paste your get_features_targets function here
def get_features_targets(data):
    features = np.stack((data['u']-data['g'],data['g']-data['r'],data['r']-data['i'],data['i']-data['z']), axis = -1)
    targets = data['redshift']
    return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))


def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)
    # predict using the model
    predictions = model.predict(test_features)
        
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions  


# complete this function
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  galaxies = data[data['spec_class'] == b'GALAXY']
  qsos = data[data['spec_class'] == b'QSO']
  # return the seperated galaxies and qsos arrays
  return galaxies, qsos

def cross_validate_predict(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return cross_validate_predictions(dtr, features, targets, 10), targets

if __name__ == "__main__":
    data = np.load('./sdss_galaxy_colors.npy')

    # Split the data set into galaxies and QSOs
    galaxies, qsos= split_galaxies_qsos(data)

    galaxies_prediction, galaxies_target = cross_validate_predict(galaxies)
    qsos_prediction, qsos_target = cross_validate_predict(qsos)

    plt.scatter(qsos_target, qsos_prediction, s=0.4, c="#e1b941")
    plt.scatter(galaxies_target, galaxies_prediction, s=0.4, c="#4169e1")
    plt.xlim((0, galaxies_target.max() if galaxies_target.max() > qsos_target.max() else qsos_target.max()))
    plt.ylim((0, galaxies_prediction.max() if galaxies_prediction.max() > qsos_prediction.max() else qsos_prediction.max()))
    plt.xlabel('Measured Redshift')
    plt.ylabel('Predicted Redshift')
    plt.show()




