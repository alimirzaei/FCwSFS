import numpy as np

# function for generating cube dataset

def gen_cube(n_features=20, data_points=20000, sigma=0.1, seed=123):
    assert n_features >= 10, 'cube data have >= 10 num of features'
    np.random.seed(seed)
    clean_points = np.random.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1,2,4]))
    points = clean_points + np.random.normal(0, sigma, (data_points, 3))
    features = np.random.rand(data_points, n_features)
    for i in range(data_points):
        offset = labels[i];
        for j in range(3):
            features[i, offset + j] = points[i, j]
    return features, labels



if __name__ == "__main__":
    data = gen_cube()
    print(data)