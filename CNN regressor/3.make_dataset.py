# Copyright 2026 Yokohama National University. All Rights Reserved.

# This code integrates and splits the session-wise feature and label data created in 2.shape_data.py by group, creating test, training, and validation datasets using a Leave-One-Group-Out (LOGO) scheme.
# The cnn_dataset folder will be created, and within it, subfolders for the dataset will also be generated.

#import public module
import numpy as np
import os
import itertools


def append_to_variable(variable, array, size):
    if variable is None: 
        variable = np.array(array).reshape(size)
    else:
        variable = np.concatenate([variable, np.array(array).reshape(size)], axis=0)
    return variable


def create_DataSet(test_date, label, model):
    """
    A function that aggregates session-level feature and label data created in 2.shape_data.py at the group level and splits them into test, training, and validation sets using a Leave-One-Group-Out (LOGO) scheme.
    """
    dates = ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"]
    sessions = ["session1F","session3F"]

    # === Validation date is automatically set to the day after test_date ===
    test_index = dates.index(test_date)
    val_date = dates[(test_index + 1) % len(dates)]   # Loop structure

    print(f">>> Test = {test_date}, Validation = {val_date}")

    all_sessions = list(itertools.product(dates, sessions))

    x_train, y_train = None, None
    x_val, y_val = None, None
    x_test, y_test = None, None

    for element_date, element_session in all_sessions:
        base_filename = element_date + element_session
        x_path1 = f'cnn_dataset/{model}/feature/x_{base_filename}_part1.npy'
        x_path2 = f'cnn_dataset/{model}/feature/x_{base_filename}_part2.npy'
        y_path1 = f'cnn_dataset/{model}/label_{label}/y_{base_filename}_part1.npy'
        y_path2 = f'cnn_dataset/{model}/label_{label}/y_{base_filename}_part2.npy'

        if not all(os.path.exists(p) for p in [x_path1, x_path2, y_path1, y_path2]):
            print(f"Missing files for {base_filename}, skipping...")
            continue

        x_tmp = np.concatenate([np.load(x_path1), np.load(x_path2)], axis=0)
        y_tmp = np.concatenate([np.load(y_path1), np.load(y_path2)], axis=0)

        # ---- Classification ----
        if element_date == test_date:
            x_test = x_tmp if x_test is None else np.concatenate([x_test, x_tmp], axis=0)
            y_test = y_tmp if y_test is None else np.concatenate([y_test, y_tmp], axis=0)

        elif element_date == val_date:
            x_val = x_tmp if x_val is None else np.concatenate([x_val, x_tmp], axis=0)
            y_val = y_tmp if y_val is None else np.concatenate([y_val, y_tmp], axis=0)

        else:
            x_train = x_tmp if x_train is None else np.concatenate([x_train, x_tmp], axis=0)
            y_train = y_tmp if y_train is None else np.concatenate([y_train, y_tmp], axis=0)

    if x_test is None or y_test is None:
        raise ValueError(f"Missing test data for {test_date}")
    if x_val is None or y_val is None:
        raise ValueError(f"Missing validation data for {val_date}")

    filename_base = test_date
    base_dir = f'cnn_dataset/{model}/dataset/recognize_label_{label}'
    # Create directories for training, test, and validation sets if they don't exist
    os.makedirs(f'{base_dir}/train/', exist_ok=True)
    os.makedirs(f'{base_dir}/test/', exist_ok=True)
    os.makedirs(f'{base_dir}/val/', exist_ok=True)

    # Print dataset size summary
    print("\n=== Dataset Size Summary ===")
    print(f"Train: x={x_train.shape}, y={y_train.shape}")
    print(f"Val:   x={x_val.shape}, y={y_val.shape}")
    print(f"Test:  x={x_test.shape}, y={y_test.shape}")

    # Save training data to .npy files
    np.save(f'{base_dir}/train/x_train_for_{filename_base}.npy', x_train)
    np.save(f'{base_dir}/train/y_train_for_{filename_base}.npy', y_train)
    # Save test data to .npy files
    np.save(f'{base_dir}/test/x_test_for_{filename_base}.npy', x_test)
    np.save(f'{base_dir}/test/y_test_for_{filename_base}.npy', y_test)
    # Save validation data to .npy files
    np.save(f'{base_dir}/val/x_val_for_{filename_base}.npy', x_val)
    np.save(f'{base_dir}/val/y_val_for_{filename_base}.npy', y_val)

    return x_train, y_train, x_val, y_val, x_test, y_test

#Main processing loop
model = "Unified_functional_spectrum_pair"
label_set = [1, 2, 3, 4, 5, 6]
test_dates = ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"]

for label in label_set:
    for test_date in test_dates:
        print(f"Processing label={label}, test_date={test_date}")
        create_DataSet(test_date, label, model)
