import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv("housing.csv")

# Remove rows with missing values
data.dropna(inplace=True)

# Split features and target
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Join training features and target
train_data = x_train.join(y_train)

# Log transformation
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# Convert ocean_proximity into dummy columns
dummies = pd.get_dummies(train_data['ocean_proximity'])
train_data = train_data.join(dummies)
train_data = train_data.drop(['ocean_proximity'], axis=1)

# Feature engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

# Final training data
x_train = train_data.drop(['median_house_value'], axis=1)
y_train = train_data['median_house_value']

# Scale training data for Linear Regression
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)

# Train Linear Regression
reg = LinearRegression()
reg.fit(x_train_s, y_train)

# Prepare test data
test_data = x_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

dummies = pd.get_dummies(test_data['ocean_proximity'])
test_data = test_data.join(dummies)
test_data = test_data.drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Final test data
x_test_processed = test_data.drop(['median_house_value'], axis=1)
y_test = test_data['median_house_value']

# Make sure test columns match train columns
x_test_processed = x_test_processed.reindex(columns=x_train.columns, fill_value=0)

# Scale test data for Linear Regression
x_test_s = scaler.transform(x_test_processed)

# Linear Regression score
print("Linear Regression score:", reg.score(x_test_s, y_test))

# Random Forest model
forest = RandomForestRegressor(random_state=42)
forest.fit(x_train, y_train)

print("Random Forest score:", forest.score(x_test_processed, y_test))

# Grid Search
param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}

grid_search = GridSearchCV(
    forest,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True
)

grid_search.fit(x_train, y_train)

best_forest = grid_search.best_estimator_

print("Best Random Forest score:", best_forest.score(x_test_processed, y_test))