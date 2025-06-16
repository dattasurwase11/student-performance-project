# train.py
import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(os.path.join(args.train, "train.csv"))
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
