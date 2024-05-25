import numpy as np
import pickle
import torch

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.data.CustomDataSet import CustomDataSet

def train_linear_regression():



    with open('../../data/processed/data_with_embaddings.pkl', 'rb') as file:
        data = pickle.load(file)

    numpy_matrix = []

    for i in range(len(data)):
        numpy_matrix.append(data[i].get_embedding().numpy())

    # Stack the NumPy arrays into a single NumPy matrix
    numpy_matrix = np.vstack(numpy_matrix)

    X_train, X_test, y_train, y_test = train_test_split([numpy_matrix for i in range(len(data))],
                                                        [data[i].cloatting for i in range(len(data))],
                                                        train_size=0.7,
                                                        shuffle=True)


    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    with open('../../models/regressor.pkl', 'wb') as file:
        pickle.dump(reg, file)

    return None

train_linear_regression()