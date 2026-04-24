# Copyright 2026 Yokohama National University. All Rights Reserved.

# This code is a program that trains a model using the dataset created in 3.make_dataset.py and then performs predictions with the trained model.
# The pred_regression_result folder will be created.

# import public module
import numpy as np
import csv
from sklearn.metrics import mean_absolute_error
# import CNN model
from model import CNN
# Import standard modules
import random
import os
import tensorflow as tf
# for handling date and time operations
import datetime

def out_pred(time,original_frame,test_frame,pred_data,test_date,model_ver,label,note=''):
    """
    The prediction results are output as text files in [all_frames, 1] format.
    
    Parameters
    ----------
    time: str
        Time of prediction execution
    original_frame: list
        All frame numbers, including those for which predictions are performed or not
    test_frame: list
        Frame numbers of the frames to be predicted
    pred_data: list
        Prediction results by the model
    label: int
        Number of bases (1–6)
    debate: str
        Meeting name / input format, e.g., "0807session1F"
    note: str
        Notes to include in the file name 
    test_date:
        Test date, e.g., 0807

    Return
    ------
    None

    """

    # Create a directory to save regression prediction results
    os.makedirs('pred_regression_result/' + model_ver + '/label' + str(label) + '/',exist_ok=True)

    # Define the output file path for prediction results
    # The file name includes test date, time, and an optional note
    path_name = f'pred_regression_result/{model_ver}/label{label}/{test_date}_{time}_{note}.txt'

    # Index for accessing prediction data
    index = 0
    # Ensure the parent directory of the output file exists
    os.makedirs(os.path.dirname(path_name), exist_ok=True)

    # Write prediction results to a text file
    with open(path_name,'w') as f:
        for j,frame_ in enumerate(original_frame): 
            frame = int(frame_)
            # If the frame index is 0, write 0 as the prediction
            if frame == 0:
                f.write('{}\t{}'.format(frame,0))

            # If the frame is included in the test frames,
            # write the corresponding predicted value
            elif frame in test_frame:
                f.write('\n{}\t{}'.format(frame,pred_data[index]))
                index += 1

            # Otherwise, write 0 for frames without prediction
            else:
                f.write('\n{}\t{}'.format(frame,0))

        f.close()    

    return None

def seed_everything(seed=42):
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.
    Also configure TensorFlow to use single-threaded execution for deterministic behavior.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

seed_everything(42)

if __name__ == "__main__":
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    data_date = None

    #Store the average values for each of the six unified functional spectrum labels (the leftmost value is fixed at 0.0).
    average = [0.0,0.45364909141196524,0.12249268221957813,0.16054819891063235,0.2900756796062207,0.11269199023463392,0.1366004136888317]
    
    model_ver = "Unified_functional_spectrum_pair"

    label_set = [1,2,3,4,5,6] #For each basis
    test_dates = ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"] #Each test group

    # The following performs training and prediction for each specified test group and specified label.
    for label in label_set:
        pred_all = []
        ans_all  = []
        mae_list = []
        for test_date in test_dates: 
            filename_base = test_date
            print(f"***************************\n target ---> {filename_base}\n***************************\n")

            base_dir = f'cnn_dataset/{model_ver}/dataset/recognize_label_{label}'

            # Load data by group (date)
            x_train = np.load(f'{base_dir}/train/x_train_for_{filename_base}.npy')
            y_train = np.load(f'{base_dir}/train/y_train_for_{filename_base}.npy')
            x_test  = np.load(f'{base_dir}/test/x_test_for_{filename_base}.npy')
            y_test  = np.load(f'{base_dir}/test/y_test_for_{filename_base}.npy')
            x_val  = np.load(f'{base_dir}/val/x_val_for_{filename_base}.npy')
            y_val  = np.load(f'{base_dir}/val/y_val_for_{filename_base}.npy')

            # A list combining one group (two sessions on the same day)
            test_sessions = [f"{test_date}session1F", f"{test_date}session3F"]

            
            def load_and_offset(session_list, base_path):
                """
                A function that, for each test_date, loads the .npy files for the two sessions and concatenates the frame numbers.
                """
                offset = 0
                merged = []
                for session in session_list:
                    data = np.load(f'{base_path}/frame/total_frame_{session}.npy', allow_pickle=True)
                    data_offset = data + offset
                    merged.append(data_offset)
                    offset = data_offset.max() + 1
                return np.concatenate(merged)

            total_frame = load_and_offset(test_sessions, f'cnn_dataset/{model_ver}')
            

            offset = 0
            merged_test_frame = []
            for session in test_sessions:
                data = np.load(f'cnn_dataset/{model_ver}/frame/test_frame_{session}.npy', allow_pickle=True)
                data_offset = data + offset
                merged_test_frame.append(data_offset)

                # Ensure the offset is added using the length of total_frame.
                total_len = len(np.load(f'cnn_dataset/{model_ver}/frame/total_frame_{session}.npy', allow_pickle=True))
                offset += total_len
            # For test_frame as well, load the .npy files for the two sessions for each test_date and concatenate the frame numbers.
            test_frame = np.concatenate(merged_test_frame)

            # Model creation, training, and prediction
            # Note: The input shape for the second set of data depends on the fusion type:
            # - For late fusion, use shape (5, 96)
            # - For early fusion, use shape (5, 1)
            cnn = CNN(
                20, (5,96), 20, (6,1), 10, (3,1), 10, 0,
                x_train, y_train, x_test, y_test, x_val, y_val,
                x_train, y_train, x_test, y_test, x_val, y_val,
                filename_base, time, label, average, model_ver
            )

            cnn.random_sampling(data_date=data_date)
            cnn.execute()
            pred = cnn.pred.reshape(-1,)

            out_pred(cnn.time, total_frame, test_frame, pred, filename_base, model_ver, label, f'label={label}')

            pred = []
            ans = []

            # A list combining one group (two sessions on the same day)
            test_sessions = [f"{test_date}session1F", f"{test_date}session3F"]

            def load_and_offset(session_list, base_path):
                """
                A function that, for each test_date, loads the .npy files for the two sessions and concatenates the frame numbers.
                """
                offset = 0
                merged = []
                for session in session_list:
                    data = np.load(f'{base_path}/frame/total_frame_{session}.npy', allow_pickle=True)
                    data_offset = data + offset
                    merged.append(data_offset)
                    offset = data_offset.max() + 1
                return np.concatenate(merged)

            total_frame = load_and_offset(test_sessions, f'cnn_dataset/{model_ver}')#全フレーム(22000とか)
            

            offset = 0
            merged_test_frame = []
            for session in test_sessions:
                data = np.load(f'cnn_dataset/{model_ver}/frame/test_frame_{session}.npy', allow_pickle=True)
                data_offset = data + offset
                merged_test_frame.append(data_offset)

                # Ensure the offset is added using the length of total_frame.
                total_len = len(np.load(f'cnn_dataset/{model_ver}/frame/total_frame_{session}.npy', allow_pickle=True))
                offset += total_len
            # For test_frame as well, load the .npy files for the two sessions for each test_date and concatenate the frame numbers.
            test_frame = np.concatenate(merged_test_frame)

            # Merge the unified functional spectrum label data and frame numbers for each group (for each test date).
            frame_org_list = []
            ans_file_list = []
            offset = 0
            for session in test_sessions:
                h_file_path = f'data/Unified_functional_spectrum_debate/H_{session}.txt'
                frames = np.loadtxt(h_file_path, usecols=[0])
                labels = np.loadtxt(h_file_path, usecols=[label])
                
                frames_shifted = frames + offset
                frame_org_list.append(frames_shifted)
                ans_file_list.append(labels)
                
                offset += frames.max() + 1

            frame_org = np.concatenate(frame_org_list)
            ans_file = np.concatenate(ans_file_list)

        
            # Prediction results
            pred_data = np.loadtxt(
                f"pred_regression_result/{model_ver}/label{label}/{test_date}_{time}_label={label}.txt",
                usecols=[1]
            )
                  
            for i,frame_ in enumerate(total_frame): 
                num = int(frame_)
                if num in test_frame:
                    pred.append(pred_data[num])
                    ans.append(ans_file[num])

                else:
                    pass
            
            # Calculate and output the MAE for each test date (each group).
            mae_group = mean_absolute_error(pred, ans)
            mae_list.append(mae_group)
            print(f"[{test_date}] MAE = {mae_group}")


            # Accumulate data to calculate the overall MAE across all groups.
            pred_all.extend(pred)
            ans_all.extend(ans)

        # Overall MAE across all groups
        global_mae = mean_absolute_error(pred_all, ans_all)
        print("=== Overall MAE ===", global_mae)

        # Create a directory for evaluation results for the current model version if it doesn't exist
        os.makedirs(f"eval/{model_ver}",exist_ok=True)
        # Define the CSV file path for storing MAE results for the current label
        csv_file = f"eval/{model_ver}/mae_0-1_singletask_label={label}.csv"
        # Check if the CSV file already exists to determine if headers should be written
        first_write = not os.path.isfile(csv_file)

        # Open the CSV file in append mode
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # If this is the first write, write the header row
            if first_write:
                writer.writerow(["date", "label", "mae_overall", "test_date", "mae_group"])
            # Write MAE results
            for i, td in enumerate(test_dates):
                writer.writerow([time, str(label), str(global_mae), td, mae_list[i]])
