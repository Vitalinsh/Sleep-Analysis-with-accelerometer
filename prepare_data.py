import numpy as np
import pandas as pd
from scipy.stats import mode

def change_labels(sample, n_sleep_stages=1):
    """
    Returns:
    sample - contains only label 1(awake) and 0(sleep) for polisomnography if n_sleep_stages == 1,
    else if n_sleep_stages == 4, then labels (5 classes): 0 - REM, 1,2,3 - no-REM sleep stages 3-1, 4 - awake,
    else if n_sleep_stages == 2, then labels (3 classes): 0 - REM, 1 - no-REM sleep stage, 2 - awake.
    """
    if sleep_stages == 1:
        # for 2 class
        sample.gt[sample.gt==0] = 8
        sample.gt[sample.gt==5] = 0
        sample.gt[np.logical_or.reduce((sample.gt==6, sample.gt==7, sample.gt==8))] = 4
    
    elif sleep_stages == 2:
        # for 3 class
        sample.gt[sample.gt==0] = 8
        sample.gt[sample.gt==5] = 0
        sample.gt[np.logical_or.reduce((sample.gt==1, sample.gt==2, sample.gt==3))] = 1
        sample.gt[np.logical_or.reduce((sample.gt==6, sample.gt==7, sample.gt==8))] = 2
    elif sleep_stages == 4:
        # for 5 class
        sample.gt[sample.gt==0] = 8
        sample.gt[np.logical_or.reduce((sample.gt==1, sample.gt==2, sample.gt==3, sample.gt==5))] = 0
        sample.gt[np.logical_or.reduce((sample.gt==6, sample.gt==7, sample.gt==8))] = 1
       
    else:
        print("Error! Wrong number of classes! Possible values: 1, 2 and 4")
    
    return sample   

#-------------------------------------------------------------------------

def decoder(sample):
    '''
    Returns: 
    decoded_sample - contains accelerometer and ps data for each sensor record, ndarray of shape (n_records, 4)
    
    '''

    sample = np.repeat(sample, sample.d, axis=0)
    n_records = sample.shape[0]
    decoded_sample = np.zeros((n_records, 4))
    
    decoded_sample[:, 0] = sample.x
    decoded_sample[:, 1] = sample.y
    decoded_sample[:, 2] = sample.z
    decoded_sample[:, 3] = sample.gt
    
    return decoded_sample

#-------------------------------------------------------------------------

def divide_by_windows(decoded_sample, window_len=60):
    """
    Parameters:
    window_len - length of each window in seconds, int
    
    Returns:
    X - accelerometer data, ndarray of shape (n_windows, window_len, 3)
    y - polisomnography data, ndarray of shape (n_windows, )
    """
    
    window_len *= 100
    n_windows = decoded_sample.shape[0] // window_len
    
    X = np.zeros((n_windows, window_len, 3))
    y = np.zeros(n_windows)
    
    for i in range(n_windows):
        X[i] = decoded_sample[window_len * i: window_len * i + window_len, 0: 3]
        
        y[i], _ = mode(decoded_sample[window_len * i: window_len * i + window_len, 3], axis=0)
                    
    return X, y

#-------------------------------------------------------------------------

def get_one_patient_data(data_path, patient, window_len=60, sleep_stages=False):
    
    """
    Returns:
    X, y - for one patient
    """
    
    sample = np.load("%s\p%s.npy"%(data_path, patient)).view(np.recarray)
    sample = change_labels(sample, sleep_stages=sleep_stages)
    sample = decoder(sample)
    X, y = divide_by_windows(sample, window_len)
    
    return X, y

#-------------------------------------------------------------------------

def get_data_for_model(data_path, patient_list, window_len=60):
    
    """
    Returns:
    X, y - for all patient in list, ndarray of shape (n_records, n_features, n_channels=3)
    """
    
    X_all_data = []
    y_all_data = []
    for patient in patient_list:
        X, y = get_one_patient_data(data_path, patient, window_len)
        X_all_data.append(X)
        y_all_data.append(y)
        
    X_all_data = np.concatenate(X_all_data, axis=0)
    y_all_data = np.concatenate(y_all_data, axis=0)
    
    return X_all_data, y_all_data

#----------------------------------------------------------------------------

def save_statistic_features(patient_list, sorce_path="ICHI14_dataset\data", save_path="features.csv", window_len=60, n_sleep_stages=1):
    
    """
    Save .csv file with extracted statistic features for each windows and axis.
    
    List of all features: ["id", "sleep_stage", "gender", "age", "std_x", "std_y", "std_z", "ptp_x", "ptp_y", "ptp_z", "mean_x", "mean_y", "mean_z", "rms_x", "rms_y", "rms_z", "crest_factor_x", "crest_factor_y", "crest_factor_z", "max_val_x", "max_val_y", "max_val_z", "min_val_x", "min_val_y", "min_val_z"]
    """
    
    columns = ["id", "sleep_stage", "gender", "age", "std_x", "std_y", "std_z", "ptp_x", "ptp_y", "ptp_z",
              "mean_x", "mean_y", "mean_z", "rms_x", "rms_y", "rms_z", "crest_factor_x", "crest_factor_y",
              "crest_factor_z", "max_val_x", "max_val_y", "max_val_z", "min_val_x", "min_val_y", "min_val_z"]
    statistics_df = pd.DataFrame(columns=columns)

    patient_data = np.load(sorce_path + '/pat_inf.npy')
    
    for patient in patient_list:
        
        X, y = get_one_patient_data(data_path=sorce_path, patient=patient, 
                                                 window_len=window_len, n_sleep_stages=n_sleep_stages)
        
        patient_id = np.array([patient] * y.shape[0]).reshape(y.shape[0], 1)
        std = np.std(X, axis=1)
        ptp = np.ptp(X, axis=1)
        mean = np.mean(X, axis=1)
        rms = np.sqrt(np.mean(np.square(X), axis=1))
        crest_factor = np.max(X, axis=1) / rms
        max_val = np.amax(X, axis=1)
        min_val = np.amin(X, axis=1)
        
        gender = 0
        age = 0
        for i, p in enumerate(patient_data[1:, 0]):
            if patient == p.decode('utf-8'):
                age = int(patient_data[i+1, 2].decode('utf-8'))
                
                if "m" == patient_data[i+1, 1].decode('utf-8'):
                    gender = 1
        age = age * np.ones((y.shape[0], 1))
        gender = gender * np.ones((y.shape[0], 1))
     
        y = y.reshape(y.shape[0], 1)
        X_new = np.concatenate((patient_id, y, gender, age, std, ptp, mean, rms, crest_factor, max_val, min_val), axis=1)
        X_new_df = pd.DataFrame(X_new, columns=columns)
        
        statistics_df = statistics_df.append(X_new_df, ignore_index=True)
        
    statistics_df.to_csv(save_path, sep=',', header=True, index=None)
    
#-----------------------------------------------------------------

def load_statistic_features(patient_list, data_path="statistic_features.csv", 
                          statistics_list=["std_x", "std_y", "std_z"]):
    
    statistics_df = pd.read_csv(data_path)
    indexes = np.logical_or.reduce([statistics_df.id == i for i in patient_list])
    
    X = statistics_df.loc[indexes, statistics_list]
    y = statistics_df.loc[indexes, "sleep_stage"]
    
    X = np.array(X)
    y = np.array(y)

    return X, y

#--------------------------------------------------------------

def load_stat_features_others_windows(patient_list, data_path="statistic_features.csv", 
                                      statistics_list=["std_x", "std_y", "std_z"], n_others_windows=40):
    """
    Returns:
    X_all_data - ndarray of shape(n_records, n_new_features), feature-vector consist of features of current window and several others ( n_others_windows // 2 before current window and n_others_windows // 2 after it)
    
    y_all_data - ndarray of shape(n_records,)
    """
    
    statistics_df = pd.read_csv(data_path)
    X_all_data = []
    y_all_data = []
    
    for patient in patient_list:
        
        X = np.array(statistics_df.loc[statistics_df.id == patient, statistics_list])
        y = np.array(statistics_df.loc[statistics_df.id == patient, "sleep_stage"])

        X_new = np.zeros((X.shape[0]-n_others_windows, X.shape[1]*(n_others_windows+1)))
        
        for i in range(0, X.shape[0]-n_others_windows):
            X_buff = X[i]
            for j in range(1, n_others_windows+1):
                X_buff = np.concatenate((X_buff, X[i+j]))
            X_new[i] = X_buff                            
    
        y = y[(n_others_windows//2): -(n_others_windows//2)]
        #y_test_new = y_test[previous:]
        
        X_all_data.append(X_new)
        y_all_data.append(y)
        
    X_all_data = np.concatenate(X_all_data, axis=0)
    y_all_data = np.concatenate(y_all_data, axis=0)
    
    return X_all_data, y_all_data

#--------------------------------------------------------------------

def load_stat_features_others_windows_rnn(patient_list, data_path="statistic_features_60s.csv", 
                                          statistics_list=["std_x", "std_y", "std_z"], n_others_windows=40):
    """  
    Returns:
    X_all_data - ndarray of shape(n_records, n_others_windows + 1, n_statistic_features), feature-vector consist of features of current window and several others ( n_others_windows // 2 before current window and n_others_windows // 2 after it)
    
    y_all_data - ndarray of shape(n_records,)
    """

    statistics_df = pd.read_csv(data_path)
    X_all_data = []
    y_all_data = []
    
    for patient in patient_list:
        
        X = np.array(statistics_df.loc[statistics_df.id == patient, statistics_list])
        y = np.array(statistics_df.loc[statistics_df.id == patient, "sleep_stage"])

        X_new = np.zeros((X.shape[0]-n_others_windows, (n_others_windows + 1), len(statistics_list)))
        
        for i in range(0, X.shape[0]-n_others_windows):
            X_buff = np.zeros((n_others_windows + 1, len(statistics_list)))
            
            for j in range(0, n_others_windows + 1):
                X_buff[j] = X[i+j]
            X_new[i] = X_buff                            
    
        y = y[(n_others_windows//2): -(n_others_windows//2)]
        #y_test_new = y_test[previous:]
        
        X_all_data.append(X_new)
        y_all_data.append(y)
        
    X_all_data = np.concatenate(X_all_data, axis=0)
    y_all_data = np.concatenate(y_all_data, axis=0)
    
    return X_all_data, y_all_data