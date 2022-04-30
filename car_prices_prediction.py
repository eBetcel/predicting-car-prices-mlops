"""
Module for car prices prediction project from DataQuest
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 99

COLS = ['symboling', 'normalized-losses',
        'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style',
        'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width',
        'height', 'curb-weight', 'engine-type',
        'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-rate', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

CARS = pd.read_csv('imports-85.data', names=COLS)

CARS.head()

# Select only the columns
#  with continuous values
#  from -
# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
CONTINOUS_VALUES_COLS = [
    'normalized-losses', 'wheel-base', 'length',
    'width', 'height', 'curb-weight',
    'engine-size', 'bore', 'stroke',
    'compression-rate', 'horsepower', 'peak-rpm',
    'city-mpg', 'highway-mpg', 'price'
    ]
NUMERIC_CARS = CARS[CONTINOUS_VALUES_COLS]

NUMERIC_CARS.head(5)

NUMERIC_CARS = NUMERIC_CARS.replace('?', np.nan)
NUMERIC_CARS.head(5)

NUMERIC_CARS = NUMERIC_CARS.astype('float')
NUMERIC_CARS.isnull().sum()

# Because `price` is the column
#  we want to predict, let's remove
#  any rows with missing `price` values.
NUMERIC_CARS = NUMERIC_CARS.dropna(subset=['price'])
NUMERIC_CARS.isnull().sum()

# Replace missing values in other columns using column means.
NUMERIC_CARS = NUMERIC_CARS.fillna(NUMERIC_CARS.mean())

# Confirm that there's no more missing values!
NUMERIC_CARS.isnull().sum()

# Normalize all columnns to range from 0 to 1 except the target column.
PRICE_COL = NUMERIC_CARS['price']
NUMERIC_CARS = (NUMERIC_CARS - NUMERIC_CARS.min())/(NUMERIC_CARS.max() - NUMERIC_CARS.min())
NUMERIC_CARS['price'] = PRICE_COL


def knn_train_test_0(train_col, target_col, data_frame):
    """
    Train model using knnn

    Args:
    train_col: list. List of columns training
    target_col: list. List of columns target
    data_frame: pandas.df. Df to be analyzed

    Return
    rmse
    """
    knn = KNeighborsRegressor()
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(data_frame.index)
    rand_data_frame = data_frame.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_data_frame) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_data_frame = rand_data_frame.iloc[0:last_train_row]
    test_data_frame = rand_data_frame.iloc[last_train_row:]

    # Fit a KNN model using default k value.
    knn.fit(train_data_frame[[train_col]], train_data_frame[target_col])

    # Make predictions using model.
    predicted_labels = knn.predict(test_data_frame[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_data_frame[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

RMSE_RESULTS = {}
TRAIN_COLS = NUMERIC_CARS.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `RMSE_RESULTS`.
for col in TRAIN_COLS:
    rmse_val = knn_train_test_0(col, 'price', NUMERIC_CARS)
    RMSE_RESULTS[col] = rmse_val

# Create a Series object from the dictionary so
# we can easily view the results, sort, etc
RMSE_RESULTS_SERIES = pd.Series(RMSE_RESULTS)
RMSE_RESULTS_SERIES.sort_values()

def knn_train_test_1(train_col, target_col, data_frame):
    """
    Train model using knnn

    Args:
    train_col: list. List of columns training
    target_col: list. List of columns target
    data_frame: pandas.df. Df to be analyzed

    Return
    rmse
    """
    np.random.seed(1)
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(data_frame.index)
    rand_data_frame = data_frame.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_data_frame) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_data_frame = rand_data_frame.iloc[0:last_train_row]
    test_data_frame = rand_data_frame.iloc[last_train_row:]

    k_values = [1, 3, 5, 7, 9]
    k_rmses = {}

    for k_1 in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_1)
        knn.fit(train_data_frame[[train_col]], train_data_frame[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_data_frame[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_data_frame[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_1] = rmse
    return k_rmses

K_RMSE_RESULTS = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `RMSE_RESULTS`.
TRAIN_COLS = NUMERIC_CARS.columns.drop('price')
for col in TRAIN_COLS:
    rmse_val = knn_train_test_1(col, 'price', NUMERIC_CARS)
    K_RMSE_RESULTS[col] = rmse_val

for k, v in K_RMSE_RESULTS.items():
    x = list(v.keys())
    y = list(v.values())

    plt.plot(x, y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

# Compute average RMSE across different `k` values for each feature.
FEATURE_AVG_RMSE = {}
for k_2, v in K_RMSE_RESULTS.items():
    avg_rmse = np.mean(list(v.values()))
    FEATURE_AVG_RMSE[k_2] = avg_rmse
SERIES_AVGE_RMSE = pd.Series(FEATURE_AVG_RMSE)
SORTED_SERIES_AVGE_RMSE = SERIES_AVGE_RMSE.sort_values()
print(SORTED_SERIES_AVGE_RMSE)

SORTED_FEATURES = SORTED_SERIES_AVGE_RMSE.index

def knn_train_test_2(train_cols, target_col, data_frame):
    """
    Train model using knnn

    Args:
    train_col: list. List of columns training
    target_col: list. List of columns target
    data_frame: pandas.df. Df to be analyzed

    Return
    rmse
    """
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(data_frame.index)
    rand_data_frame = data_frame.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_data_frame) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_data_frame = rand_data_frame.iloc[0:last_train_row]
    test_data_frame = rand_data_frame.iloc[last_train_row:]

    k_values = [5]
    k_rmses = {}

    for k_3 in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_3)
        knn.fit(train_data_frame[train_cols], train_data_frame[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_data_frame[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_data_frame[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_3] = rmse
    return k_rmses

K_RMSE_RESULTS = {}

for nr_best_feats in range(2, 7):
    K_RMSE_RESULTS['{} best features'.format(nr_best_feats)] = knn_train_test_2(
        SORTED_FEATURES[:nr_best_feats],
        'price',
        NUMERIC_CARS
    )

def knn_train_test3(train_cols, target_col, data_frame):
    """
    Train model using knnn

    Args:
    train_col: list. List of columns training
    target_col: list. List of columns target
    data_frame: pandas.df. Df to be analyzed

    Return
    rmse
    """
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(data_frame.index)
    rand_data_frame = data_frame.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_data_frame) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_data_frame = rand_data_frame.iloc[0:last_train_row]
    test_data_frame = rand_data_frame.iloc[last_train_row:]

    k_rmses = {}

    for k_4 in range(1, 25):
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_4)
        knn.fit(train_data_frame[train_cols], train_data_frame[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_data_frame[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_data_frame[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_4] = rmse
    return k_rmses

K_RMSE_RESULTS = {}

for nr_best_feats in range(2, 6):
    K_RMSE_RESULTS['{} best features'.format(nr_best_feats)] = knn_train_test3(
        SORTED_FEATURES[:nr_best_feats],
        'price',
        NUMERIC_CARS
    )

for k, v in K_RMSE_RESULTS.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x, y, label="{}".format(k))

plt.xlabel('k value')
plt.ylabel('RMSE')
plt.legend()
