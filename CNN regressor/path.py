# Copyright 2026 Yokohama National University. All Rights Reserved.

# This code specifies the file paths for the input features and the unified functional spectrum label data used in the preprocessing scripts 1-1 and 1-2.

# session name
session = ['2024_0807session1F','2024_0807session3F',
           '2024_0809session1F','2024_0809session3F',
           '2024_0826session1F','2024_0826session3F',
           '2024_0829session1F','2024_0829session3F',
           '2024_0905session1F','2024_0905session3F',
           '2024_0910session1F','2024_0910session3F'] 
# participant name
P       = ['P1','P2','P3','P4'] 

# Pair name of speaker, addressee, and other listeners
# Speaker–addressee order is ignored (e.g., P1_P2_P3P4 == P2_P1_P3P4)
Pair = ['P1_P2_P3P4','P1_P3_P2P4','P1_P4_P2P3','P2_P3_P1P4','P2_P4_P1P3','P3_P4_P1P2']

# Input feature paths
head_pose_angle_ = ['data/head_pose_angle_data/','.openface.txt']    # Head pose angle path
utter_           = ['data/utter_data/','.utter.txt']            # Utterance activity path
gaze_status_GT_  = ['data/gaze_status_GT_data/','.gaze.status_GT.txt'] # Gaze state (ground truth) path
eyeball_         = ['data/eyeball_data/','.OpenFace.EyeforCNN.txt']  # Gaze direction path
au_              = ['data/au_data/','.OpenFace.corrFrame.AU_norm.','.txt']  # Facial expression AU data path

# Data paths for conditional branching
floor_holder_ = ['data/FloorHolder_data/','.FloorHolder.mainlabeler.txt'] # Floor holder path
gaze_direction_ = ['data/gaze_direction_data/','.gaze_label.withTransition.txt'] # Gaze direction path

# Unified Functional Spectrum label data paths
Unified_functional_spectrum_pair_ = ['data/Unified_functional_spectrum_pair/H_','.txt'] # Path for each participant pair
Unified_functional_spectrum_debate_ = ['data/Unified_functional_spectrum_debate/H_','.txt'] # Path for each debate (date)
ave_Unified_functional_spectrum_pair_ = ['data/Unified_functional_spectrum_pair/ave_','.txt'] # Path storing the average label values

head_pose_angle        = {}
utter                  = {}
gaze_status_GT         = {}
eyeball                = {}
au                     = {}
floor_holder           = {}
Unified_functional_spectrum_pair     = {}
ave_Unified_functional_spectrum_pair = {}
Unified_functional_spectrum_debate        = {}
gaze_direction         = {}

# Detailed path settings for head pose angles, utterance status, gaze status predictions,
# eyeball directions, AU features, unified functional spectrum label data,
# and data used for conditional branching
for debate in session:
    head_pose_angle[debate] = head_pose_angle_[0] + debate + head_pose_angle_[1]
    utter[debate]           = utter_[0] + debate + utter_[1]
    gaze_status_GT[debate]  = gaze_status_GT_[0] + debate + gaze_status_GT_[1]
    eyeball[debate]         = eyeball_[0] + debate + eyeball_[1]
    floor_holder[debate]    = floor_holder_[0] + debate + floor_holder_[1]
    gaze_direction[debate]  = gaze_direction_[0] + debate + gaze_direction_[1]
    Unified_functional_spectrum_debate[debate] = Unified_functional_spectrum_debate_[0] + debate + Unified_functional_spectrum_debate_[1]
    for person in P:
        au[debate + person] = au_[0] + debate + au_[1] + person + au_[2]
    for pair in Pair:
        Unified_functional_spectrum_pair[debate + pair]     = Unified_functional_spectrum_pair_[0] + debate + pair + Unified_functional_spectrum_pair_[1]
        ave_Unified_functional_spectrum_pair[debate + pair] = ave_Unified_functional_spectrum_pair_[0] + debate + pair + ave_Unified_functional_spectrum_pair_[1]

# Save path
save       = 'processed_data/'
save_TF     = ['false/','true/']