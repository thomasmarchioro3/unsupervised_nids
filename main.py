import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports 
from data.utils import load_dataset
from src.preprocess import NIDS_Preprocessor
from src.nids.autoencoder import NIDS_AutoEncoder

def seed_all(seed: int=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_errors(errors, y, calibration_threshold: float=None):
    
    bins = np.linspace(np.min(errors), np.percentile(errors, 99), 100)

    plt.figure(figsize=(10, 5))
    plt.hist(errors[y == 0], bins=bins, alpha=0.5, density=True, label='Normal')
    plt.hist(errors[y == 1], bins=bins, alpha=0.5, density=True, label='Anomaly')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')

    if calibration_threshold is not None:    
        plt.axvline(x=calibration_threshold, color='k', linestyle='--', label='Calibration Threshold')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    dataset_name = 'nslkdd'
    test_split = 0.2
    calibration_split = 0.2  # Percentage of training data to use for calibration
    use_subsample = False  # Used to test the code
    subsample_size = 0.01

    seed_all(42)

    X, y = load_dataset(dataset_name, use_subsample, subsample_size)
    y = y.apply(lambda x: 0 if x == 'normal' else 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)
    # Keep only normal data for training
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=calibration_split, shuffle=False)

    preprocessor = NIDS_Preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_cal = preprocessor.transform(X_cal)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.astype(np.float32).to_numpy()
    X_cal = X_cal.astype(np.float32).to_numpy()
    X_test = X_test.astype(np.float32).to_numpy()

    autoencoder = NIDS_AutoEncoder(n_features=X_train.shape[1], noise_stddev=0.0, calibration_strategy='z_score', z_threshold=3.0)
    autoencoder.fit(X_train)
    autoencoder.calibrate(X_cal)

    errors = autoencoder.evaluate_errors(X_test)
    y_pred = autoencoder.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    plot_errors(errors, y_test, autoencoder.threshold)