# importing Libs

# scikit-learn
from sklearn.preprocessing import StandardScaler # KNN should have scaled datas. High diffrence between datas of all columns and rows will cause more error rate
from sklearn.neighbors import KNeighborsRegressor # KNN-Regression Model
from sklearn.model_selection import train_test_split # Split some of dataset for train and some for test
from sklearn.metrics import mean_squared_error # Calculate the error rate

# pandas
import pandas as pd

# load data.csv file.
# we use CSV for our datasets
data = pd.read_csv("data.csv")

# the columns that we want to train model
x = data[["area","bedrooms","year","parking","elevator"]]
# feature
y = data[["price"]]

"""
x = data that we know -------------------------------|
y = result and feature--------------------------|    |
train = the part that themodel should know    <-|  <-|
test = the part that the model should predict <-|  <-|
"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=.8, random_state=42
)

# scale dataset values
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# define the model
model = KNeighborsRegressor(n_neighbors=3)
# train the model
model.fit(x_train, y_train)

# predict test part of data.csv
y_pred = model.predict(x_test)
print(y_pred)

# calculate the error rate
mse = mean_squared_error(y_test, y_pred)
print(100 - mse)
