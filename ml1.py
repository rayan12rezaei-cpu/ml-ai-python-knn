from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.read_csv("data.csv")

x = data[["area","bedrooms","year","parking","elevator"]]
y = data[["price"]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=.8, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(100 - mse)