import numpy as np
from sklearn.tree import DecisionTreeRegressor

# copy in your get_features_targets function here
def get_features_targets(data):
    features = np.stack((data['u']-data['g'],data['g']-data['r'],data['r']-data['i'],data['i']-data['z']), axis = -1)
    targets = data['redshift']
    return features, targets

def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))

def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  split = features.shape[0]//2
  train_features = features[:split]
  train_targets  = targets[:split]
  test_features = features[split:]
  test_targets  = targets[split:]
  # train the model
  model.fit(train_features, train_targets)
  # get the predicted_redshifts
  predictions = model.predict(test_features)
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)

# load the data and generate the features and targets
data = np.load('sdss_galaxy_colors.npy')
features, targets = get_features_targets(data)
  
# initialize model

dtr = DecisionTreeRegressor()

# train the model

dtr.fit(features, targets)

# make predictions using the same features

predictions = dtr.predict(features)


# print out the first 4 predicted redshifts

print(targets[:4])
print(predictions[:4])

print(median_diff(predictions, targets))

## Model with train and test datasets
diff = validate_model(dtr, features, targets)
print('Median difference: {:f}'.format(diff))
