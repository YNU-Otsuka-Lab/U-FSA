# Copyright 2026 Yokohama National University. All Rights Reserved.

# This code extracts unified functional spectrum label (intensity) data for each label and saves them as NumPy (.npy) files.
# Inside the processed_data folder, folders Unified_functional_spectrum_pair_label_1 to Unified_functional_spectrum_pair_label_6 will be created.

#import public module
import numpy as np
import os

#import private module
import path as DATAPATH

class Dataset():
    """
    A class that stores time-series intensity data of the interaction functional spectrum as class variables, implemented in two versions: per pair and per session.

    Attributes
    ----------
    debate : str
        Name of the target debate.
    pair : str
        Target pair ID (P1_P2,P1_P3,P1_P4,P2_P3,P2_P4,P3_P4)
    basis: int
        Target basis ID (1,2,...,6)
    Unified_functional_spectrum_pair_path : str
        File path of the unified functional spectrum data for each pair.
        Pair name of speaker, addressee, and other listeners
        Speaker–addressee order is ignored (e.g., P1_P2_P3P4 == P2_P1_P3P4)
    Unified_functional_spectrum_debate_path : str
        File path of the unified functional spectrum data for each debate (date)
    label_(1~6) : ndarray
        Data array of unified functional spectrum labels (intensity)

    Methods
    -------
    load_label :
        Store the intensity of each basis as labels from the unified functional spectrum (intensity) data file.

    """
    def __init__(self,debate,pair,basis):
        """
        Parameters
        ----------
        debate: str
            Name of the target debate.
        pair: str
            Target pair ID (P1_P2,P1_P3,P1_P4,P2_P3,P2_P4,P3_P4)
        basis: int
            Target basis ID (1,2,...,K)
        """
        # A set of identifiers.
        self.debate = debate
        self.pair   = pair
        self.basis  = basis

        #Path to the label data file
        self.Unified_functional_spectrum_pair_path = DATAPATH.Unified_functional_spectrum_pair[debate + pair]
        self.Unified_functional_spectrum_debate_path  = DATAPATH.Unified_functional_spectrum_debate[debate]

        # Path to the data used for conditional branching
        self.floorholder_path    = DATAPATH.floor_holder[debate]
        self.gaze_direction_path = DATAPATH.gaze_direction[debate]

        #Class variable to store the extracted data
        # Pair-wise labels
        self.Unified_functional_spectrum_pair_label = []
        # Debate-wise labels
        self.Unified_functional_spectrum_debate_label = []

    def load_Unified_functional_spectrum_pair_label(self):
        """
        Store the intensity of the specified unified functional spectrum for the specified pair in the label.
        """
        _label_ = np.loadtxt(self.Unified_functional_spectrum_pair_path)
        #Extract only specific categories from the set of categories.
        for frame_data in _label_:
            self.Unified_functional_spectrum_pair_label.append(frame_data[self.basis])
        self.Unified_functional_spectrum_pair_label = np.array(self.Unified_functional_spectrum_pair_label)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + f'Unified_functional_spectrum_pair_label_{self.basis}/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + f'Unified_functional_spectrum_pair_label_{self.basis}/'+ self.debate + self.pair,self.Unified_functional_spectrum_pair_label)

    def load_Unified_functional_spectrum_debate_label(self):
        """
        Store the intensity of the unified functional spectrum for a specified debate in the label.
        """
        _label_ = np.loadtxt(self.Unified_functional_spectrum_debate_path)
        #Extract only specific categories from the set of categories.
        for frame_data in _label_:
            self.Unified_functional_spectrum_debate_label.append(frame_data[self.basis])
        self.Unified_functional_spectrum_debate_label = np.array(self.Unified_functional_spectrum_debate_label)
        # Create a directory to save data in NPY format.
        os.makedirs(DATAPATH.save + f'Unified_functional_spectrum_debate_label_{self.basis}/',exist_ok = True)
        # Save the data in NPY format to avoid re-running the preprocessing.
        np.save(DATAPATH.save + f'Unified_functional_spectrum_debate_label_{self.basis}/'+ self.debate ,self.Unified_functional_spectrum_debate_label)

    def load_data(self):
        """
        Store the time-series intensity data of the unified functional spectrum in a class variable.
        If the corresponding npy file already exists, load the data from that file.
        If the file does not exist, create the data for all samples by extracting it from the raw data.
        """
        try:
            #Input features
            #Label data
            self.Unified_functional_spectrum_pair_label = np.load(DATAPATH.save + f'Unified_functional_spectrum_pair_label_{self.basis}/'+self.debate + self.pair +'.npy',allow_pickle=True)
            self.Unified_functional_spectrum_debate_label = np.load(DATAPATH.save + f'Unified_functional_spectrum_debate_label_{self.basis}/'+self.debate +'.npy',allow_pickle=True)

        except:
        #Class variable to store the extracted data
        #Label data
            #Unified_functional_spectrum_pair
            self.Unified_functional_spectrum_pair_label = []
            self.Unified_functional_spectrum_debate_label    = []

            self.load_Unified_functional_spectrum_pair_label()
            self.load_Unified_functional_spectrum_debate_label()


for date in ["2024_0807","2024_0809","2024_0826","2024_0829","2024_0905","2024_0910"]:
    for session in ["session1F","session3F"]:
        for pair in ["P1_P2_P3P4","P1_P3_P2P4","P1_P4_P2P3","P2_P3_P1P4","P2_P4_P1P3","P3_P4_P1P2"]:
            for basis in range(1,7):
                debate = date + session
                print("now_loading...",debate+pair+f'basis{basis}')
                load_data = Dataset(debate,pair,basis)
                load_data.load_data()
