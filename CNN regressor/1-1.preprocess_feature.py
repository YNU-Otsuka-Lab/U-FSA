# Copyright 2026 Yokohama National University. All Rights Reserved.

# This code performs preprocessing on the input features.
# The processed_data folders will be created.
# It extracts the behavioral data (head pose angular velocity, utterance status, gaze status, eyeball direction, and facial expressions) and performs preprocessing. 
# Except for the utterance status, the input features are obtained from CSV files extracted using OpenFace.
# The head pose angular velocities were calculated by the frame difference of the head pose angles between adjacent frames. They are clipped to [-200, 200] and normalized to the range between -1 and 1. 
# The utterance status is the temporally smoothed sequence using a moving average window (50 frames long) applied to a binary sequence indicating the framewise presence or absence of an utterance obtained from a manual transcript. 
# The eyeball direction is defined as the horizontal angle of the eyeball relative to the frontal facial direction. This value is computed from the OpenFace CSV outputs, clipped to the range [−20, 20], and normalized to [−1, 1].
# Facial expressions were the intensities of 17 action units and were normalized to the range between 0 and 1.

#import public module
import numpy as np
import os

#import private module
import path as DATAPATH

class Dataset():
    """
    A class that stores each feature dataset as class variables.  

    Attributes
    ----------
    debate : str
        Name of the target debate.
    person : str
        Target participant ID (P1, P2, P3, P4).
    index : int
        Index of the participant ID (P1=0, P2=1, P3=2, P4=3), used for head_pose_angle and eyeball.
    head_pose_angle_path : str
        Path to the head pose motion data file.
    utter_path : str
        Path to the utterance activity data file.
    gaze_status_GT_path : str
        Path to the gaze status (ground truth) data file.
    eyeball_path : str
        Path to the eyeball direction data file.
    AU_path : str
        Path to the facial expression data file.
    floorholder_path : str
        Path to the floorholder data file.
    gaze_direction_path : str
        Path to the gaze direction data file.

    head_pose_angle : ndarray
        Array of head pose angular velocity data.
    utter : ndarray
        Array of utterance activity data.
    gaze_status_GT : ndarray
        Array of gaze status (ground truth).
    eyeball : ndarray
        Array of eyeball direction data.
    au : ndarray
        Array of facial expression data (action units).
    floorholder : ndarray
        Array of floorholder data.
    gaze_direction : ndarray
        Array of gaze direction data.

    Methods
    -------
    load_head_pose_angle :
        Extracts head pose angles from the head pose data file, computes the head pose angular velocity, and stores the result in head_pose_angle.
    load_utter :
        Extracts binary utterance activity from the utterance data file, applies smoothing, and stores the result in utter.
    load_gaze_status_GT :
        Extracts gaze status from the gaze status data file and stores the result in gaze_status_GT.
    load_eyeball:
        Extracts eyeball direction from the eyeball data file, applies normalization, and stores the result in eyeball.
    load_AU :
        Extracts action unit (AU) values from the facial expression data and stores the result in au.
    load_floorholder :
        Extracts floorholder information from the floorholder data and stores the result in floorholder.
    load_gaze_direction :
        Extracts gaze direction from the gaze direction data and stores the result in gaze_direction.
    load_data :
        Store time-series data of head motion angular velocity (azimuth, elevation, roll), utterance activity, gaze status, eyeball direction, and facial expressions as class variables.
        Load data from existing NPY files if available; otherwise, generate the data by extracting and preprocessing all datasets.
    """

    def __init__(self,debate,person):
        """
        Parameters
        ----------
        debate: str
            Target debate name.
        person: str
            Target participant ID (P1, P2, P3, P4).
        """
        #A set of identifiers.
        self.debate = debate
        self.person = person
        self.index = ["P1","P2","P3","P4"].index(self.person)
        #Path to the input feature file.
        self.head_pose_angle_path = DATAPATH.head_pose_angle[debate]
        self.utter_path           = DATAPATH.utter[debate]
        self.gaze_status_GT_path  = DATAPATH.gaze_status_GT[debate]
        self.eyeball_path         = DATAPATH.eyeball[debate]
        self.AU_path              = DATAPATH.au[debate + person] 
        #Path to the data used for conditional branching
        self.floorholder_path     = DATAPATH.floor_holder[debate]
        self.gaze_direction_path  = DATAPATH.gaze_direction[debate]

        #Class variables for storing the extracted data.
        #Input features
        self.head_pose_angle = []
        self.utter           = []
        self.gaze_status_GT  = []
        self.eyeball         = []
        self.au              = []
        #Data for conditional branching
        self.floorholder     = []
        self.gaze_direction  =[]

    def load_head_pose_angle(self):
        """
        Extracts head pose angles from the head pose data file, computes the head pose angular velocity, and stores the result in head_pose_angle.
        """
        # Indices of the head shaking, nodding, and tilting directions in the head_pose_angle data.
        AZIMUTH_INDEX   = 6
        ELEVATION_INDEX = 7
        ROLL_INDEX      = 8
        # Adjust the starting index according to the participant ID.
        index = 7 * self.index
        indexs = [AZIMUTH_INDEX + index, ELEVATION_INDEX + index, ROLL_INDEX + index]
        _head_pose_angle_ = np.genfromtxt(self.head_pose_angle_path, delimiter=",", dtype=float, autostrip=True)  
        for i,frame_data in enumerate(_head_pose_angle_):
            # The initial value is defined as [0, 0, 0] because an accurate velocity cannot be obtained from the initial angular coordinates alone.
            if i == 0:
                self.head_pose_angle = np.append(self.head_pose_angle,[0,0,0])
            # Compute the angular velocity from the difference between consecutive coordinate data.
            else:
                tmp = []
                for j in range(len(frame_data)):
                    if j in indexs:
                        #Angular velocity calculation
                        tmp_head_pose_angle = (_head_pose_angle_[i][j] - _head_pose_angle_[i - 1][j])*29.97
                        #Normalize the range [-200, 200] to [-1, 1].
                        if tmp_head_pose_angle>=200:
                            tmp_head_pose_angle=1
                        elif tmp_head_pose_angle<=-200:
                            tmp_head_pose_angle=-1
                        elif -200<tmp_head_pose_angle<200:
                            tmp_head_pose_angle=tmp_head_pose_angle/200
                        tmp = np.append(tmp,tmp_head_pose_angle)
                self.head_pose_angle = np.append(self.head_pose_angle,tmp)
        self.head_pose_angle = self.head_pose_angle.reshape(-1,3)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'head_pose_angle/',exist_ok=True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'head_pose_angle/' + self.debate + self.person,self.head_pose_angle)

    def load_utter(self):
        """
        Extracts binary utterance activity from the utterance data file, applies smoothing, and stores the result in utter.
        """
        # Adjust the starting index according to the participant ID.
        index = 2 + self.index
        _utter_ = np.loadtxt(self.utter_path)
        tmp = []
        for frame_data in _utter_:
            tmp.append(frame_data[index])
        # Smoothing (moving average with a window size of 101; 50 frames before and after).
        filter = np.ones(101) / 101
        tmp = np.array(tmp)
        self.utter = np.convolve(tmp,filter,mode = 'same')# * 2 - 1
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'utter/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'utter/' + self.debate + self.person,self.utter)

    def load_gaze_status_GT(self):
        """
        Extracts gaze status from the gaze status data file and stores the result in gaze_status_GT.
        """
        # Adjust the starting index according to the participant ID.
        # 追加
        index = 2 + self.index
        _gaze_status_GT_ = np.loadtxt(self.gaze_status_GT_path)
        tmp = []
        for frame_data in _gaze_status_GT_:
            tmp.append(frame_data[index])
        self.gaze_status_GT = np.array(tmp)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'gaze_status_GT/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'gaze_status_GT/' + self.debate + self.person,self.gaze_status_GT)

    def load_eyeball(self):
        """
        Extracts eyeball direction from the eyeball data file, applies normalization, and stores the result in eyeball.
        """
        # Indices of the horizontal direction data for the left eye(x_left),and right eye(x_right)in the eyeball data.
        X_left_INDEX = 2
        X_right_INDEX = 3
        # Adjust the starting index according to the participant ID.
        # 追加
        index = 3 * self.index
        indexs = [X_left_INDEX + index, X_right_INDEX + index]
        _eyeball_ = np.loadtxt(self.eyeball_path)
        for frame_data in _eyeball_:
            tmp=[]
            #Normalize the range [-20, 20] to [-1, 1].
            for i in indexs:
                if frame_data[i]>=20:
                    tmp_eyeball = 1
                elif frame_data[i]<=-20:
                    tmp_eyeball = -1
                elif -20<frame_data[i]<20:
                    tmp_eyeball = frame_data[i]/20
                tmp = np.append(tmp,tmp_eyeball)
            self.eyeball = np.append(self.eyeball,tmp)
        self.eyeball = self.eyeball.reshape(-1,2)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'eyeball/',exist_ok=True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'eyeball/' + self.debate + self.person,self.eyeball)

    def load_AU(self):
        """
        Extracts action unit (AU) values from the facial expression data and stores the result in au.
        """
        # Load action unit (AU) values from the text data.
        # arg = 17
        _au_ = np.loadtxt(self.AU_path)
        
        for i,frame_data in enumerate(_au_):
            # An array for temporarily storing one column of AU values.
            tmp = []
            # Elements from the 3rd to the 19th columns of the AU data are appended sequentially.
            for j in range(3,20):                         
                tmp = np.append(tmp,_au_[i][j])
            self.au = np.append(self.au,tmp)
        self.au = np.array(self.au)
        self.au = self.au.reshape(-1,17)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'AU/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'AU/' + self.debate + self.person,self.au)

    # Data for conditional branching
    def load_floorholder(self):
        """
        Extracts floorholder information from the floorholder data and stores the result in floorholder.
        """
        # Adjust the starting index according to the participant ID.
        # 追加
        index = 2 + self.index
        _floorholder_ = np.loadtxt(self.floorholder_path)
        tmp = []
        for frame_data in _floorholder_:
            tmp.append(frame_data[index])
        self.floorholder = np.array(tmp)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'FloorHolder/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'FloorHolder/' + self.debate + self.person,self.floorholder)

    def load_gaze_direction(self):
        """
        Extracts gaze direction from the gaze direction data and stores the result in gaze_direction.
        """
        # Adjust the starting index according to the participant ID.
        index = 2 + self.index
        _gaze_direction_ = np.loadtxt(self.gaze_direction_path)
        tmp = []
        for frame_data in _gaze_direction_:
            tmp.append(frame_data[index])
        self.gaze_direction = np.array(tmp)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + 'gaze_direction/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + 'gaze_direction/' + self.debate + self.person,self.gaze_direction)

    def load_data(self):
        """
        Store time-series data of head motion angular velocity (azimuth, elevation, roll), utterance activity, gaze status, eyeball direction, and facial expressions as class variables.
        Load data from existing NPY files if available; otherwise, generate the data by extracting and preprocessing all datasets.
        """
        try:
            #Input features
            self.head_pose_angle = np.load(DATAPATH.save + 'head_pose_angle/' + self.debate + self.person + '.npy',allow_pickle=True)
            self.utter           = np.load(DATAPATH.save + 'utter/'           + self.debate + self.person + '.npy',allow_pickle=True)
            self.gaze_status_GT  = np.load(DATAPATH.save + 'gaze_status_GT/'  + self.debate + self.person + '.npy',allow_pickle=True)
            self.eyeball         = np.load(DATAPATH.save + 'eyeball/'         + self.debate + self.person + '.npy',allow_pickle=True)
            self.au              = np.load(DATAPATH.save + 'AU/'              + self.debate + self.person + '.npy',allow_pickle=True)
            #Data for conditional branching
            self.floorholder     = np.load(DATAPATH.save + 'FloorHolder/'     + self.debate + self.person + '.npy',allow_pickle=True)
            self.gaze_direction  = np.load(DATAPATH.save + 'gaze_direction/' + self.debate + self.person + '.npy',allow_pickle=True)

        except:
        #Class variables for storing the extracted data.
        #Input features
            self.head_pose_angle  = []
            self.utter            = []
            self.gaze_status_GT   = []
            self.eyeball          = []
            self.au               = []
        #Data for conditional branching
            self.floorholder      = []
            self.gaze_direction = []

            self.load_head_pose_angle()
            self.load_utter()
            self.load_gaze_status_GT()
            self.load_eyeball()
            self.load_AU()
            self.load_floorholder()
            self.load_gaze_direction()


for group in ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"]:
    for session in ["session1F","session3F"]:
        for person in ["P1","P2","P3","P4"]:
            debate = group + session
            print("now_loading...",debate+person)
            ds = Dataset(debate,person)
            ds.load_data()
