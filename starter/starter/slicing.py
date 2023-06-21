import sys
import os
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
from constants import cat_features

def slice_metrics(model, data, slice_feature, categorical_features=[]):
    """
    Output the performance of the model on slices of the data

    Inputs
    ------
    model : any
        Trained machine learning model.
    data : pd.DataFrame
        Dataframe containing the features and label.
    slice_feature: str
        Name of the feature used to make slices (categorical features)
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    Returns
    -------
    None
    """
    original_stdout = sys.stdout
    with open(os.path.join(os.path.dirname(__file__), "slice_output.txt"), "w") as f:
        sys.stdout = f
        print("Slicing data based on", slice_feature)
        print("====================================")
        X, y, _, _ = process_data(
            data, categorical_features=categorical_features, label="salary", training=True
        )
        preds = inference(model, X)

        for slice_value in data[slice_feature].unique():
            slice_index = data.index[data[slice_feature] == slice_value]
            
            print(slice_feature, '=', slice_value)
            print('data size:', len(slice_index))
            print('precision: {}, recall: {}, fbeta: {}'.format(
                *compute_model_metrics(y[slice_index], preds[slice_index])
            ))
            print('-------------------------------------------------')
        sys.stdout = original_stdout


if __name__ == '__main__':
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, '../data/clean_cencus.csv'))

    model_path = os.path.join(file_dir, '../model/rf_model.pkl')
    loaded_model = pickle.load(open(model_path, 'rb'))

    slice_metrics(loaded_model, data, 'education', categorical_features=cat_features)