from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class FCwSFS:
    def __init__(self, distance_threshold = 0.5):
        self.selector = SelectKBest(score_func=f_classif, k=1)
        self.distance_threshold = distance_threshold
    
    # it is lazy approach so we only normalize the training data durring training phase
    def fit(self, X_train, y_train):  
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        self.X_train = self.scaler.transform(X_train)
        print(self.X_train)
        self.y_train = y_train
        self.num_classes = len(set(y_train))
    
    def predict(self, X_test, max_features = 5, certainty_threshold = 1):
        y_predicted = []
        all_selected_features = []
        X_test_scaled = self.scaler.transform(X_test)
        for x_t in tqdm(X_test_scaled):
            selected_faetures = []
            predicted_y = []
            X_train_sub = self.X_train.copy()
            y_train_sub = self.y_train.copy()
            for f in range(max_features):
                if(len(y_train_sub) == 0):
                    break
                X_train_sub[:, selected_faetures] = 0 # this is because the selected feature would not be selected in next phase
                self.selector.fit_transform(X_train_sub, y_train_sub)
                next_feature = self.selector.get_support(indices=True)[0]
                selected_faetures.append(next_feature)
                
                diff = X_train_sub[:, selected_faetures[-1:]] - x_t[selected_faetures[-1:]]
                distances = np.linalg.norm(diff, axis = 1)
                selected_samples = distances < self.distance_threshold
                X_train_sub = X_train_sub[selected_samples, :]
                y_train_sub = y_train_sub[selected_samples]
                probs = np.bincount(y_train_sub, minlength=self.num_classes) / len(y_train_sub)
                predicted_y.append(np.argmax(probs))
                if(np.max(probs)  > certainty_threshold):
                    break
                # print(y_train_sub)
            # print("selected_features", selected_faetures)
            all_selected_features.append(selected_faetures)
            y_predicted.append(predicted_y[-1])
        return np.array(y_predicted), all_selected_features
    


    

