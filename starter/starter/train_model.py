# Script to train machine learning model.
import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from constants import cat_features
# Add the necessary imports for the starter code.

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.
rf_model = train_model(X_train, y_train)

model_path = os.path.join(file_dir, '../model/rf_model.pkl')
pickle.dump(rf_model, open(model_path, 'wb'))

encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
pickle.dump(encoder, open(encoder_path, 'wb'))

lb_path = os.path.join(file_dir, '../model/lb.pkl')
pickle.dump(lb, open(lb_path, 'wb'))

# Predict
preds = inference(rf_model, X_test)

print('precision: {}, recall: {}, fbeta: {}'.format(
    *compute_model_metrics(y_test, preds)
))