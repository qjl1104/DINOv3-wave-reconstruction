# utils/file_io.py
import pickle
import numpy as np
import os

def save_data_pickle(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")

def load_data_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def save_calibration_params(params_dict, filepath): # Not used by calibrate.py directly, but can be a utility
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, **params_dict)
    print(f"Calibration parameters saved to {filepath}")

def load_calibration_params(filepath): # Example utility
    try:
        data = np.load(filepath)
        print(f"Calibration parameters loaded from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: Calibration file {filepath} not found.")
        return None