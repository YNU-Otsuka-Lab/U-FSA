# Copyright 2026 Yokohama National University. All Rights Reserved.

# This is a program used in 4.training_and_prediction.py that contains the model configuration.

# import public module
## random
import random
## np
import numpy as np
## os
import os
## csv
import csv

### layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
### models
from keras.models import Model

## tensorflow module
import tensorflow as tf
### optimizers
from tensorflow.keras import optimizers
### L2 norm
from tensorflow.keras.regularizers import l2

class CNN():
    def __init__(self,filter1,kernel1_size,filter2,kernel2_size,filter3,kernel3_size,dense_unit,dropout_rate,x_train,y_train,x_test,y_test,x_val,y_val,x_train_org,y_train_org,x_test_org,y_test_org,x_val_org,y_val_org,test_date,time,label,average,model_ver):
        """
        CNN model built using Keras

        Parameters
        ----------
        filter1-3: int
            Number of filters in the 1st–3rd convolutional layers
        kernel1-3_size: tuple (x,y)
            Kernel size of the 1st–3rd convolutional layers
        dense_unit: int
            Number of units in the 4th fully connected layer
        dropout_rate: float
            Dropout rate of the dropout layer
        x_train, x_train_org: array
            Feature array of the training data
        y_train, y_train_org: array
            Label array of the training data
        x_val, x_val_org: array
            Feature array of the validation data
        y_val, y_val_org: array
            Label array of the validation data
        x_test, x_test_org: array
            Feature array of the test data
        y_test, y_test_org: array
            Label array of the test data
        test_date: str
            Target dates for prediction (0807, 0809, 0826, 0829, 0905, 0910)
        time: str
            Experiment start time
        average: list
            Mean value of the data to be predicted
        model_ver: str
            Model name
        """
        # Experiment start time
        self.time = time

        # Model name
        self.model_ver = model_ver
        
        # Mean value of the data to be predicted
        self.average = average
        
        # Predicted labels
        self.label = label

        # Model name
        self.model_name = "CNN"

        # Target for estimation
        self.date = test_date

        # Configuration of training, validation, and test data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        # Keep the original data
        self.x_train_org = x_train_org
        self.y_train_org = y_train_org   
        self.x_val_org = x_val_org
        self.y_val_org = y_val_org 
        self.x_test_org = x_test_org
        self.y_test_org = y_test_org    
                
        # Size of input features
        self.input_shape = (32,96,1)

        # Number and size of convolutional kernels
        self.filter1 = filter1
        self.kernel1_size = kernel1_size
        self.filter2 = filter2
        self.kernel2_size = kernel2_size
        self.filter3 = filter3
        self.kernel3_size = kernel3_size

        # Kernel size of pooling layers
        self.pool_size = (4,1)
        self.pool_strides = (2,1)

        # Number of units in fully connected layers
        self.dense_unit = dense_unit

        # Dropout rate
        self.dropout_rate = dropout_rate

        # Optimization method
        self.learning_rate = 1.0
        self.momentum = 0
        # Setting the name of the optimization method
        self.optimizer_name = "Adadelta"
        # Configure the actual optimizer according to the specified optimizer name
        if self.optimizer_name == "SGD":
            self.optimizer = optimizers.SGD(learning_rate=self.learning_rate,momentum=self.momentum,name="SGD")
        elif self.optimizer_name == "Adam":
            self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            self.momentum = None
        elif self.optimizer_name == "Adadelta":
            self.optimizer = optimizers.Adadelta(learning_rate=self.learning_rate)
            self.momentum = None
        elif self.optimizer_name == "Adagrad":
            self.optimizer = optimizers.Adagrad(learning_rate=self.learning_rate)
            self.momentum = None
        else:
            print("Error: This optimzer is not defined.")
            self.optimzer = None

        # Setting the loss function name
        self.loss_name = "mae"
        self.loss = "mae"
        self.metrics = "mae" 
        self.metrics_name = "mae"

        # Training parameters
        self.batch_size = 12
        self.epochs = 100
        self.epoch_patient = 10

        # Whether to use early stopping
        self.early_stopping_switch = True

        # Parameters
        self.parameter_dict = {"examine_time":self.time,
                            "filter1":self.filter1,
                            "filter2":self.filter2,
                            "filter3":self.filter3,
                            "kernel1_size":self.kernel1_size,
                            "kernel2_size":self.kernel2_size,
                            "kernel3_size":self.kernel3_size,
                            "pool_size":self.pool_size,
                            "pool_stride":self.pool_strides,
                            "dense_unit":self.dense_unit,
                            "dropout_rate":self.dropout_rate,
                            "optimizer":self.optimizer_name,
                            "learning_rate":self.learning_rate,
                            "momentum":self.momentum,
                            "loss":self.loss_name,
                            "batch_size":self.batch_size,
                            "epoch":self.epochs,
                            "early_stopping":self.early_stopping_switch,
                            "epoch_patient":self.epoch_patient
                            }

    def random_sampling(self,data_date = None):
        """
        Sample data for each mean value level based on the number of training data samples.
        # If the number of samples exceeds the reference, perform undersampling.
        # If the number of samples is below the reference, perform oversampling.
        # Oversampling is done by repeatedly performing sampling without replacement to ensure that all original data are used at least once.
        """
        if data_date == None:
            
            # Store the average value of the target label in ave.
            ave = self.average[self.label]
            
            print(ave)
            
            tmp_x = []
            tmp_y = []
            
            over_ave_x = []
            over_ave_y = []
            under_ave_x = []
            under_ave_y = []
            
            add_data_x = []
            add_data_y = []
            
            over_count = 0
            under_count = 0
                        
            # Classify samples as above the average (over_ave) or below the average (under_ave), and count the number of samples in each group.
            for i in range(len(self.y_train_org)):
                if self.y_train_org[i] >= ave:
                    over_ave_x.append(self.x_train_org[i])
                    over_ave_y.append(self.y_train_org[i])
                    over_count += 1
                else:
                    under_ave_x.append(self.x_train_org[i])
                    under_ave_y.append(self.y_train_org[i])
                    under_count += 1
            
            print(self.date)
            print("Number of samples above the average:",over_count)
            print("Number of samples below the average:",under_count)
            
            under_ave_x = np.array(under_ave_x).reshape(-1,32,96,1)
            under_ave_y = np.array(under_ave_y).reshape(-1,)
            over_ave_x = np.array(over_ave_x).reshape(-1,32,96,1)
            over_ave_y = np.array(over_ave_y).reshape(-1,)
            
            if over_count >= under_count: # When there are more samples above the average:
                # Select the same number of below-average samples as the number of above-average samples.
                # Combine over_ave and under_ave.
                for i in range(over_count):
                    num_o = i % over_count
                    num_u = i % under_count
                    
                    add_data_x.append(under_ave_x[num_u])
                    add_data_x.append(over_ave_x[num_o])
                    add_data_y.append(under_ave_y[num_u])
                    add_data_y.append(over_ave_y[num_o])
            else: # When there are more samples below the average:
                # Select the same number of above-average samples as the number of below-average samples.
                # Combine over_ave and under_ave.
                for i in range(under_count):
                    num_o = i % over_count
                    num_u = i % under_count
                    
                    add_data_x.append(under_ave_x[num_u])
                    add_data_x.append(over_ave_x[num_o])
                    add_data_y.append(under_ave_y[num_u])
                    add_data_y.append(over_ave_y[num_o])
            
            add_data_x = np.array(add_data_x).reshape(-1,32,96,1)
            add_data_y = np.array(add_data_y).reshape(-1,)
            
            # Reorder the data created above and create a list of indices.
            index = list(range(len(add_data_y)))

            # Shuffle the list of indices.
            index_shuffle = random.sample(index,len(index))
            
            for i in index_shuffle:
                tmp_x.append(add_data_x[i])
                tmp_y.append(add_data_y[i])
            
            self.x_train = np.array(tmp_x).reshape(-1,32,96,1)
            self.y_train = np.array(tmp_y).reshape(-1,)
            
            #------------------------------------------------------------------------------------------------
            
            # Create a directory to save randomly sampled training data if it does not exist
            os.makedirs(f"TrainData/{self.model_ver}/{self.time}/random_sampling/",exist_ok=True)
            # Save randomly sampled training feature data
            np.save(f"TrainData/{self.model_ver}/{self.time}/random_sampling/{self.date}_x_{self.label}",self.x_train)
            # Save randomly sampled training label data
            np.save(f"TrainData/{self.model_ver}/{self.time}/random_sampling/{self.date}_y_{self.label}",self.y_train)
            # Notify that random sampling and shuffling have been completed
            print("Shuffled data by random sampling...")

        else:
            try:
                # Attempt to load previously saved randomly sampled training data
                print("Load start....")
                print(f"TrainData/{self.model_ver}/{data_date}/random_sampling/{self.date}_x.npy")
                # Load training feature data
                self.x_train = np.load(f"TrainData/{self.model_ver}/{data_date}/random_sampling/{self.date}_x_{self.label}.npy",allow_pickle=True)
                print("x ok")
                # Load training label data
                self.y_train = np.load(f"TrainData/{self.model_ver}/{data_date}/random_sampling/{self.date}_y_{self.label}.npy",allow_pickle=True)
                print("y ok")
                # Log successful loading of the data
                print("load {} data ---- save date={}".format(self.date,data_date))
            except:
                # If loading fails, perform random sampling again
                self.random_sampling()

    def create_model(self):
        """
        Function to build the CNN model
        """
        # Input layer
        inputs = Input(shape=self.input_shape,name="inputs")
        # Convolutional and pooling layers (1st layer)
        conv1 = Conv2D(filters=self.filter1,
                        kernel_size=self.kernel1_size,
                        activation="relu",
                        kernel_regularizer=l2(0.001),
                        name="conv1")(inputs)
        pool1 = MaxPooling2D(pool_size=self.pool_size,
                        strides=self.pool_strides,
                        name="pool1")(conv1)
        norm1 = BatchNormalization(name="norm1")(pool1)
        drop1 = Dropout(0.1)(norm1) # prevent overfitting

        # Convolutional and pooling layers (2nd layer)
        conv2 = Conv2D(filters=self.filter2,
                        kernel_size=self.kernel2_size,
                        activation="relu",
                        kernel_regularizer=l2(0.001),
                        name="conv2")(drop1)
        pool2 = MaxPooling2D(pool_size=self.pool_size,
                        strides=self.pool_strides,
                        name="pool2")(conv2)
        norm2 = BatchNormalization(name="norm2")(pool2)
        drop2 = Dropout(0.2)(norm2)

        # Convolutional layer (3rd layer)
        conv3 = Conv2D(filters=self.filter3,
                        kernel_size=self.kernel3_size,
                        activation="relu",
                        kernel_regularizer=l2(0.001),
                        name="conv3")(drop2)
        norm3 = BatchNormalization(name="norm3")(conv3)
        drop3 = Dropout(0.25)(norm3)

        # Fully connected layer
        flat = Flatten(name="flat")(drop3)
        dense1 = Dense(units=self.dense_unit,
                        activation="relu",
                        kernel_regularizer=l2(0.001),
                        name="dense1")(flat)
        norm4 = BatchNormalization(name="relu")(dense1)

        # Dropout layer
        drop4 = Dropout(rate = self.dropout_rate)(norm4)

        # Output layer
        dense2 = Dense(1,activation="sigmoid",
                        kernel_regularizer=l2(0.001),
                        name="out")(drop4)
        # Build CNN model
        self.cnn = Model(inputs = inputs, outputs = dense2)

        self.cnn.compile(optimizer=self.optimizer,loss = self.loss,metrics=self.metrics)


    def predict(self):
        """
        Function to train the CNN model and perform prediction using the trained CNN model
        """
        # Create directory for saving training logs (CSVLogger)
        os.makedirs(f"CSVLogger/{self.model_ver}/{self.model_name}/{self.label}/{self.time}/",exist_ok=True)
        # CSV logger to record training history (loss, metrics) for each epoch
        csv_logger = tf.keras.callbacks.CSVLogger(f"CSVLogger/{self.model_ver}/{self.model_name}/{self.label}/{self.time}/{self.date}_({self.filter1}_{self.filter2}_{self.filter3}_{self.dense_unit}.csv",)
        # Model checkpoint callback to save the model at each epoch
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'modeldata/{self.model_ver}/label={self.label}/{self.time}/{self.date}_epoch='+'{epoch:02d}.hdf5',
                    save_weights_only=False,
                    monitor='mean_out_loss',
                    verbose=0,
                    save_best_only=False,
                    save_freq='epoch')

        # === Training phase ===
        if self.early_stopping_switch:
            # Early stopping callback to prevent overfitting
            early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=self.epoch_patient, verbose=0, mode="min", restore_best_weights=True)
            # Train the CNN model with early stopping
            self.fit_result = self.cnn.fit(x = self.x_train,y = self.y_train,                
                                            batch_size=self.batch_size,
                                            epochs=self.epochs,
                                            verbose=1,
                                            shuffle=False,
                                            validation_data=(self.x_val,self.y_val),         
                                            callbacks=[early_stopping,csv_logger,model_checkpoint_callback])
            # Display EarlyStopping configuration
            print(f"[EarlyStopping] monitor = 'val_loss', min_delta = 0.001, patience = {self.epoch_patient}, mode = 'min'")
        else:
            # Train the CNN model without early stopping
            self.fit_result = self.cnn.fit(x = self.x_train,y = self.y_train,
                                            batch_size=self.batch_size,
                                            epochs=self.epochs,
                                            verbose=1,
                                            shuffle=False,
                                            validation_data=(self.x_val,self.y_val),          
                                            callbacks=[csv_logger])         
        # === Prediction phase ===
        # Predict unified functional spectrum values for the test data
        self.pred = self.cnn.predict(self.x_test)
        # Evaluate the trained model on the test data
        self.eval = self.cnn.evaluate(self.x_test, self.y_test)


    def output_csv(self):
        """
        Output evaluation results to a CSV file
        """
        # Create directory for evaluation results if it does not exist
        os.makedirs(f"eval/{self.model_ver}/{self.model_name}/{self.label}/",exist_ok=True)

        # Create an empty CSV file if it does not already exist
        if not os.path.isfile(f"eval/{self.model_ver}/{self.model_name}/{self.label}/{self.date}.csv"):
            with open(f"eval/{self.model_ver}/{self.model_name}/{self.label}/{self.date}.csv","w",newline="") as f:
                pass
        
        # Append evaluation results to the CSV file
        with open(f"eval/{self.model_ver}/{self.model_name}/{self.label}/{self.date}.csv","a",newline="") as f:
            writer = csv.writer(f)
            # Write header row (evaluation metrics)
            writer.writerow(["time",self.loss_name,self.metrics_name,"additonal_information"])
            # Write evaluation results
            writer.writerow([self.time]+self.eval) 

    def execute(self):
        """
        Execute the full pipeline:
        model creation, training & prediction, and result output
        """
        # Build the CNN model
        self.create_model()
        # Train the model and perform prediction
        self.predict()
        # Output evaluation results to CSV
        self.output_csv()