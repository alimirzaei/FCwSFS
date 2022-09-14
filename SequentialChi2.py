from utils import get_partial_zeros
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


class SequentialChi2:
    def __init__(self) -> None:
        self.classifier = MLPClassifier(alpha=1, max_iter=1000)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, max_selected_features_percent=0.1, N=10000):
        (X_partial, y_partial, _) = get_partial_zeros(X_train, y_train, max_selected_features_percent=max_selected_features_percent, N=N)
        self.scaler.fit(X_partial)
        X_partial_normalized = self.scaler.transform(X_partial)
        # Training partial classifier
        print("Training classifier ...")
        self.classifier.fit(X_partial_normalized,y_partial)
        print("Training is done!")
        # Feature ranking for each class
        num_classes = np.max(y_train) + 1
        feature_scores = []
        print("Feature scores training ...")
        for c in range(num_classes):
            feature_ranking_model = SelectKBest(chi2).fit(X_train-np.min(X_train), y_train==c)
            feature_scores.append(feature_ranking_model.scores_)
        self.feature_scores = np.array(feature_scores)
        print("Feature score training done!")


    def test(self, X_test, y_test, max_features = 5):
        num_features = X_test.shape[1]
        y_pred = []
        for x_, y_ in zip(X_test, y_test):
            temp_feature_scores = self.feature_scores.copy()
            selected_features = []
            mask = np.zeros(num_features)
            for i in range(max_features+1):
                observed_x = x_.copy()
                observed_x[mask==0] = 0
                normalized = self.scaler.transform([np.hstack([observed_x, mask])])
                logits = self.classifier.predict_proba(normalized)
                
                next_feature = np.argmax(logits.dot(temp_feature_scores))
                selected_features.append(next_feature)
                mask[next_feature]=1
                temp_feature_scores[:, next_feature] = 0  # this feature is no longer usefull
            y_pred.append(np.argmax(logits))
        return sum(y_pred == y_test) / len(y_test)
                
        





        

