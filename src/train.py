from joblib import dump
from pathlib import Path

from dvclive import Live
import yaml

import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier

ROOT = Path('../')
PARAMS = ROOT / 'src' / 'params.yaml'


def load_images(data_frame, column_name):
    filelist = data_frame[column_name].to_list()
    image_list = imread_collection(filelist)
    return image_list


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    return label_list


def preprocess(image):
    resized = resize(image, (100, 100, 3))
    reshaped = resized.reshape((1, 30000))
    return reshaped


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    raw_images = load_images(data_frame=df, column_name="filename")
    processed_images = [preprocess(image) for image in raw_images]
    data = np.concatenate(processed_images, axis=0)
    return data, labels


def main(repo_path):
    train_csv_path = repo_path / "data/prepared/train.csv"
    train_data, labels = load_data(train_csv_path)
    with open(PARAMS, 'r') as params_yaml:
        try:
            params = yaml.safe_load()
        except yaml.YAMLError as e:
            print(e)
    with Live(save_dvc_exp=True) as live:
        live.log_param("model", params['MODEL'])
        live.log_param("epochs", params['EPOCHS'])
        if params['MODEL'] == 'SGD':
            sgd = SGDClassifier(max_iter=params['EPOCHS'])
            trained_model = sgd.fit(train_data, labels)
        out_model_path = repo_path / "model/model.joblib"
        dump(trained_model, out_model_path)
        live.log_artifact(out_model_path, type="model", name=f'{params["MODEL"]}_{params["EPOCHS"]}_epochs')


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
