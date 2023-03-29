from FCwSFS import FCwSFS
from datasets import get_syntethic_dataset
import numpy as np
import warnings

warnings.filterwarnings('ignore')


X_train, y_train = get_syntethic_dataset(5000)
X_test, y_test = get_syntethic_dataset(100)


model = FCwSFS(distance_threshold=0.3)

model.fit(X_train=X_train, y_train=y_train)

y_pred, features = model.predict(X_test, max_features=10, certainty_threshold=0.8)
print("ACCURACY", np.sum(y_pred == y_test) / len(y_test))
print("y_pred", y_pred)
print("y_pred", y_test)
print("features", features)