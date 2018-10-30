class Detector:
    
    def __init__(self, frequency=100):
        
        import pickle
        
        self.frequency = frequency;
        self.model = pickle.load(open("log_reg1.sav", 'rb'))
        
    def divide_by_windows_std(self, X, y, window_len=60):
        """
        Parameters:
        window_len - length of each window in seconds, int
        
        Returns:
        X - accelerometer data, ndarray of shape (n_windows, window_len, 3)
        y - polisomnography data, ndarray of shape (n_windows, )
        """
        from scipy.stats import mode
        import numpy as np
        
        window_len *= self.frequency
        n_windows = X.shape[0] // window_len
        
        X_new = np.zeros((n_windows, 3))
        y_new = np.zeros(n_windows)
        
        for i in range(n_windows):
            X_new[i] = np.std(X[window_len * i: window_len * i + window_len, :], axis=0)
            y_new[i], _ = mode(y[window_len * i: window_len * i + window_len], axis=0)
                        
        return X_new, y_new

    
    def combine_windows(self, X, y, n_others_windows=32):
   
        import numpy as np
        
        X_new = np.zeros((X.shape[0]-n_others_windows, X.shape[1]*(n_others_windows+1)))
        
        for i in range(0, X.shape[0]-n_others_windows):
            X_buff = X[i]
            for j in range(1, n_others_windows+1):
                X_buff = np.concatenate((X_buff, X[i+j]))
            X_new[i] = X_buff                            
    
        y = y[(n_others_windows//2): -(n_others_windows//2)]
        #y_test_new = y_test[previous:]
    
        return X_new, y
    
    def predict(self, X):
        
        y_predict = self.model.predict(X)
        
        return y_predict
        

    