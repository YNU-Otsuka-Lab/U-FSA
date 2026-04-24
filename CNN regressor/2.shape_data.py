# Copyright 2026 Yokohama National University. All Rights Reserved.

# This is a program that shapes the behavioral data into inputs for the regression model as short time-series segments of approximately 1 second (32 frames), centered on the target frame to be predicted.
# Generates the cnn_dataset folder containing subfolders for feature, label_1 to label_6, and frame.


#import public module
import numpy as np
import os
import itertools

def create_cnn_Data_label(debate,window_size,label,number_of_interlocutors,N):
    # Generates the feature datasets (for CNN input) of the speaker, addressee, and other listeners for a specified debate, along with the corresponding unified functional spectrum label datasets.
    """
    window_size : int   
        Number of frames used as a single input data (window size for feature extraction, i.e., 32 frames)
    number_of_interlocutors : int   
        Number of interlocutors
    N : int 
        Time window for averaging
    frame_forward : int
        Frame width in the future direction from the center when including multiple frames in one data sample
    frame_back : int
        Frame width in the past direction from the center when including multiple frames in one data sample
    frame_win : list(int)
        Index array of frame numbers to include when including multiple frames in one data sample
    label : int
        Number of bases (1–6)
    floorholder : int
        The floor holder (i.e., the speaker) is labeled as 800, and all others are labeled as −1.
    gaze_direction : int
        Label indicating whom the participant is looking at.
        0 corresponds to P1, 1 to P2, 2 to P3, and 3 to P4.

    Final outputs      
    frame 
        Create 24 .npy files for all sessions (6 groups × 2 sessions per group = 12 sessions) containing the test frames and total frames (frame folder).
    Feature data
        Create x_data_debate_part1 and x_data_debate_part2 (shape: number of samples × 32 × 96 × 1) for all sessions (6 groups × 2 sessions per group = 12 sessions), resulting in a total of 24 x-label files (feature folder).
        Note that part1 and part2 correspond to swapping the order of the other two listeners.
    Label data
        Create y_data_debate_part1 and y_data_debate_part2 (shape: number of samples × 1) for each label 1–6, for all sessions (6 groups × 2 sessions per group = 12 sessions), resulting in a total of 24 y-label files (label 1–6 folders).
        Note that part1 and part2 correspond to swapping the order of the other two listeners.
    """ 
    # Frame range for analysis
    frame_span = {'2024_0807session1F':[4452,24836],'2024_0807session3F':[3105,20313],
                  '2024_0809session1F':[4174,23565],'2024_0809session3F':[3395,25038],
                  '2024_0826session1F':[4320,23311],'2024_0826session3F':[3648,22231],
                  '2024_0829session1F':[4440,24180],'2024_0829session3F':[3671,24835],
                  '2024_0905session1F':[4825,26339],'2024_0905session3F':[3708,17524],
                  '2024_0910session1F':[4650,21481],'2024_0910session3F':[3658,20479]} 
    
    frame_start = frame_span[debate][0]
    frame_end = frame_span[debate][1]
    #Person ID
    person_num = {f'P{person}':person for person in range(1,number_of_interlocutors+1)}
    #Specified window width
    frame_forward = window_size // 2 + 1
    if window_size % 2 ==0:
        frame_back = - (window_size // 2) + 1
    else:
        frame_back = - (window_size // 2)
    frame_win = np.arange(frame_back,frame_forward)

    #Data loading
    #FloorHolder
    floorholder={person : np. load('processed_data/FloorHolder/' + debate + person + '.npy') for person in person_num}
    #gaze_direction
    gaze_direction = {person : np. load('processed_data/gaze_direction/' + debate + person + '.npy') for person in person_num}
    #head_pose_angle
    head_pose_angle = {person : np. load('processed_data/head_pose_angle/' + debate + person + '.npy') for person in person_num}
    #utter
    utter = {person : np. load('processed_data/utter/' + debate + person + '.npy') for person in person_num}
    #gaze_status_GT
    gaze_status_GT = {person : np. load('processed_data/gaze_status_GT/' + debate + person + '.npy') for person in person_num}
    #eyeball_direction
    eyeball = {person : np. load('processed_data/eyeball/' + debate + person + '.npy') for person in person_num}
    #AU
    au = {person : np. load('processed_data/au/' + debate + person + '.npy') for person in person_num}


    #Label data
    Unified_functional_spectrum_pair_label={}
    for pair in itertools.combinations(person_num.keys(),2):
        pair=list(pair)
        others=[p for p in person_num.keys() if p not in pair]
        Unified_functional_spectrum_pair_label['_'.join(pair)+'_'+''.join(others)]=np.load('processed_data/Unified_functional_spectrum_pair_label_' + str(label) + '/' + debate + pair[0] + '_' + pair[1] + '_' + ''.join(others) + '.npy')

    # Feature and label data
    x_data_debate_part1 = []
    x_data_debate_part2 = []
    y_data_debate_part1 = []
    y_data_debate_part2 = []

    # List storing the frame numbers of the target debate
    total_frame_debate = list(range(len(Unified_functional_spectrum_pair_label['P1_P2_P3P4'])))

    # List storing the frame numbers of the test set
    test_frame_debate = []

    min_len=min(min([len(_) for _ in head_pose_angle.values()]),
                min([len(_) for _ in utter.values()]),
                min([len(_) for _ in gaze_status_GT.values()]),
                min([len(_) for _ in eyeball.values()]),
                min([len(_) for _ in au.values()]))
    
    # List storing speaker–addressee pairs
    speaker_gaze_match = []

    for frame in range(frame_start, frame_end + 1):
        # Whether the window centered on a frame falls within a valid range
        if (frame + frame_win[0] >= 0) and (frame + frame_win[-1] < min_len):
            if floorholder['P1'][frame] == 800: #When P1 is the Floor Holder (speaker)
                if gaze_direction['P1'][frame] == person_num['P2']-1: #When P1 is looking at P2
                    speaker_gaze_match.append(12) 
                elif gaze_direction['P1'][frame] == person_num['P3']-1: #When P1 is looking at P3
                    speaker_gaze_match.append(13) 
                elif gaze_direction['P1'][frame] == person_num['P4']-1: #When P1 is looking at P4
                    speaker_gaze_match.append(14)  
                else: #When not looking at anyone
                    speaker_gaze_match.append(10) 
            elif floorholder['P2'][frame] == 800: #When P2 is the Floor Holder (speaker)
                if gaze_direction['P2'][frame] == person_num['P1']-1: #When P2 is looking at P1
                    speaker_gaze_match.append(21) 
                elif gaze_direction['P2'][frame] == person_num['P3']-1: #When P2 is looking at P3
                    speaker_gaze_match.append(23) 
                elif gaze_direction['P2'][frame] == person_num['P4']-1: #When P2 is looking at P4
                    speaker_gaze_match.append(24)  
                else: #When not looking at anyone
                    speaker_gaze_match.append(20)  
            elif floorholder['P3'][frame] == 800: #When P3 is the Floor Holder (speaker)
                if gaze_direction['P3'][frame] == person_num['P1']-1: #When P3 is looking at P1
                        speaker_gaze_match.append(31)  
                elif gaze_direction['P3'][frame] == person_num['P2']-1: #When P3 is looking at P2
                        speaker_gaze_match.append(32)  
                elif gaze_direction['P3'][frame] == person_num['P4']-1: #When P3 is looking at P4
                        speaker_gaze_match.append(34)  
                else: #When not looking at anyone
                    speaker_gaze_match.append(30) 
            elif floorholder['P4'][frame] == 800: #When P4 is the Floor Holder (speaker)
                if gaze_direction['P4'][frame] == person_num['P1']-1: #When P4 is looking at P1
                        speaker_gaze_match.append(41)  
                elif gaze_direction['P4'][frame] == person_num['P2']-1: #When P4 is looking at P2
                        speaker_gaze_match.append(42)  
                elif gaze_direction['P4'][frame] == person_num['P3']-1: #When P4 is looking at P3
                        speaker_gaze_match.append(43)  
                else: #When not looking at anyone
                    speaker_gaze_match.append(40)       
            else: #When there is no speaker
                speaker_gaze_match.append(00)

    speaker_gaze_match = np.array(speaker_gaze_match)

    label_dict={}
    for pair in itertools.combinations(person_num.keys(),2):
        pair=list(pair)
        others=[p for p in person_num.keys() if p not in pair]
        label_dict[person_num[pair[0]]*10+person_num[pair[1]]]='_'.join(pair)+'_'+''.join(others)
        label_dict[person_num[pair[1]]*10+person_num[pair[0]]]='_'.join(pair)+'_'+''.join(others)
    
    # Track the current consecutive values
    current_value = None
    current_sequence = []

    # Processing loop
    for i, value in enumerate(speaker_gaze_match):
        if value == current_value:  # In case of consecutive values
            current_sequence.append(i)
        else:  # When a new value appears
            # Check if there are at least N frames and whether the consecutive value is not 00, 10, 20, 30, or 40 (values indicating Avert)
            if len(current_sequence) >= N:
                if current_value in label_dict:  # Case 1: For the corresponding pair
                    # ---- (A) Retrieve the label array corresponding to current_value ----
                    label_array = Unified_functional_spectrum_pair_label[label_dict[current_value]]
                    # ---- (B) Process num_to_extract frames starting from the beginning of the consecutive frames ----
                    num_to_extract = len(current_sequence) - (N-1)
                    extract_indices = current_sequence[:num_to_extract]

                    for seq_idx in extract_indices:
                        target_frame = frame_start + seq_idx  # Actual frame numbers
                        # Save the frame numbers
                        test_frame_debate.append(target_frame)

                        # ---- (C) Obtain the correct ground-truth labels ----
                        y_label = label_array[target_frame]

                        # ---- (D) Feature extraction ----
                        # Inside the speaker_gaze_match loop
                        speaker = current_value // 10
                        listener = current_value % 10
                        others = [p for p in range(1,number_of_interlocutors+1) if p not in [speaker, listener]]
                        part1_order = [speaker, listener] + others
                        part2_order = [speaker, listener] + others[::-1]
                        # --- part1 ---
                        for win in frame_win:
                            idx = target_frame + win
                            for p in part1_order:
                                x_data_debate_part1.extend(head_pose_angle[f'P{p}'][idx][:])
                                x_data_debate_part1.append(utter[f'P{p}'][idx])
                                x_data_debate_part1.append(gaze_status_GT[f'P{p}'][idx])
                                x_data_debate_part1.extend(eyeball[f'P{p}'][idx][:])
                                x_data_debate_part1.extend(au[f'P{p}'][idx][:])

                        y_data_debate_part1.append(y_label)

                        # --- part2 ---
                        for win in frame_win:
                            idx = target_frame + win
                            for p in part2_order:
                                x_data_debate_part2.extend(head_pose_angle[f'P{p}'][idx][:])
                                x_data_debate_part2.append(utter[f'P{p}'][idx])
                                x_data_debate_part2.append(gaze_status_GT[f'P{p}'][idx])
                                x_data_debate_part2.extend(eyeball[f'P{p}'][idx][:])
                                x_data_debate_part2.extend(au[f'P{p}'][idx][:])

                        y_data_debate_part2.append(y_label)

            # Update to the new value
            current_value = value
            current_sequence = [i]

    # Process the last sequence
    if len(current_sequence) >= N:
                if current_value in label_dict:  # Case 1: For the corresponding pair
                    # ---- (A) Retrieve the label array corresponding to current_value ----
                    label_array = Unified_functional_spectrum_pair_label[label_dict[current_value]]
                    # ---- (B) Process num_to_extract frames starting from the beginning of the consecutive frames ----
                    num_to_extract = len(current_sequence) - (N-1)
                    extract_indices = current_sequence[:num_to_extract]

                    for seq_idx in extract_indices:
                        target_frame = frame_start + seq_idx  # Actual frame numbers
                        test_frame_debate.append(target_frame)

                        # ---- (C) Obtain the correct ground-truth labels ----
                        y_label = label_array[target_frame]

                        # ---- (D) Feature extraction ----
                        # Inside the speaker_gaze_match loop
                        speaker = current_value // 10
                        listener = current_value % 10
                        others = [p for p in range(1,number_of_interlocutors+1) if p not in [speaker, listener]]
                        part1_order = [speaker, listener] + others
                        part2_order = [speaker, listener] + others[::-1]
                        # part1
                        for win in frame_win:
                            idx = target_frame + win
                            for p in part1_order:
                                x_data_debate_part1.extend(head_pose_angle[f'P{p}'][idx][:])
                                x_data_debate_part1.append(utter[f'P{p}'][idx])
                                x_data_debate_part1.append(gaze_status_GT[f'P{p}'][idx])
                                x_data_debate_part1.extend(eyeball[f'P{p}'][idx][:])
                                x_data_debate_part1.extend(au[f'P{p}'][idx][:])

                        y_data_debate_part1.append(y_label)

                        # --- part2 ---
                        for win in frame_win:
                            idx = target_frame + win
                            for p in part2_order:
                                x_data_debate_part2.extend(head_pose_angle[f'P{p}'][idx][:])
                                x_data_debate_part2.append(utter[f'P{p}'][idx])
                                x_data_debate_part2.append(gaze_status_GT[f'P{p}'][idx])
                                x_data_debate_part2.extend(eyeball[f'P{p}'][idx][:])
                                x_data_debate_part2.extend(au[f'P{p}'][idx][:])

                        y_data_debate_part2.append(y_label)

    x_data_debate_part1 = np.array(x_data_debate_part1).reshape(-1,window_size,24*number_of_interlocutors,1)
    x_data_debate_part2 = np.array(x_data_debate_part2).reshape(-1,window_size,24*number_of_interlocutors,1)
    y_data_debate_part1 = np.array(y_data_debate_part1).reshape(-1,1)
    y_data_debate_part2 = np.array(y_data_debate_part2).reshape(-1,1)

    os.makedirs('cnn_dataset/Unified_functional_spectrum_pair/feature/',exist_ok=True)
    os.makedirs('cnn_dataset/Unified_functional_spectrum_pair/label_' + str(label) + '/',exist_ok=True)
    os.makedirs('cnn_dataset/Unified_functional_spectrum_pair/frame/',exist_ok=True)

    #Save the created feature and label data → Create the training dataset for the deep learning model
    np.save('cnn_dataset/Unified_functional_spectrum_pair/feature/x_' + debate + '_part1.npy', x_data_debate_part1)
    np.save('cnn_dataset/Unified_functional_spectrum_pair/feature/x_' + debate + '_part2.npy', x_data_debate_part2)
    np.save('cnn_dataset/Unified_functional_spectrum_pair/label_' + str(label) + '/y_' + debate + '_part1.npy', y_data_debate_part1)
    np.save('cnn_dataset/Unified_functional_spectrum_pair/label_' + str(label) + '/y_' + debate + '_part2.npy', y_data_debate_part2)
    np.save('cnn_dataset/Unified_functional_spectrum_pair/frame/total_frame_' + debate + '.npy', total_frame_debate)
    np.save('cnn_dataset/Unified_functional_spectrum_pair/frame/test_frame_' + debate + '.npy', test_frame_debate)

#Main processing loop
label_set = [1, 2, 3, 4, 5, 6]

for label in label_set:
    for date in ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"]:
        for session in ["session1F","session3F"]:
            debate = date + session

            #Function call
            create_cnn_Data_label(
                debate=debate,
                window_size=32,
                label=label,
                number_of_interlocutors=4,
                N=7
            )