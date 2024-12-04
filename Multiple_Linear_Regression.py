# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')

"""# *Sleep Efficiency* - **Multiple Linear Regression**
"""

# Import necessary libraries
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# The path is read and then assigned the variable sleep_better
sleep_better = pd.read_csv("/content/drive/MyDrive/Sleep_Efficiency.csv")
print(sleep_better)

# Print the columns for the variable sleep_better
print(sleep_better.columns)

def date_formatting(dataframe3):
# Converting 'Wakeup time' to datetime
  dataframe3['Wakeup time'] = pd.to_datetime(dataframe3['Wakeup time'], format="%Y-%m-%d %H:%M:%S")

# Converting 'Bedtime' to datetime
  dataframe3['Bedtime'] = pd.to_datetime(dataframe3['Bedtime'], format='%Y-%m-%d %H:%M:%S')
  return dataframe3

# The date_formatting fcn is called with sleep_better, and the
# modified dataframe is assigned to the variable sleep_better2
sleep_better2 = date_formatting(sleep_better)

smoking = sleep_better['Smoking status']
# Replacing 'Yes'/ 'No' to 1/0
smoking_st = smoking.replace({'Yes': 1, 'No': 0})
print(smoking_st)

import pandas as pd
#sleep_better2 = pd.concat([sleep_update, smoking_st], axis=1)
sleep_update = pd.concat([sleep_better2, smoking_st], axis=1)
print(sleep_update)

sleep_update.drop('Smoking status', axis=1, inplace=True)
print(sleep_update)

# Drops the following columns from the dataset
sleep_update = sleep_better.drop(['ID', 'Gender', 'Smoking status'], axis=1)
print(sleep_update)

# New column is created in the sleep_update dataframe
sleep_update['Smoking_Status'] = smoking_st
print(sleep_update)

# Obtaining the correlation for the dataframe sleep_update
sleep_update.corr()

"""# *Visualizing the correlation between the original dataset*
"""

# Restate the correlation matrix for the primary dataframe, sleep_update
correlation_matrix = sleep_update.corr()
# Create a heatmap
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.show()

# The following columns are dropped from the sleep_update dataframe
sleep_update2 = sleep_update.drop(['Age', 'Bedtime', 'Wakeup time', 'Sleep duration', 'REM sleep percentage', 'Caffeine consumption'], axis=1)
print(sleep_update2)

"""# *Visualizing the correlation of the updated dataset*"""

# Restate the correlation matrix for the updated dataframe, sleep_update2
corr_matrix = sleep_update2.corr()
# Create a Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Drop rows with any NaN values in either X or Y
sleep_update2 = sleep_update2.dropna()

# Separate features (independent variables) and target (dependent variable)
X = sleep_update2.drop('Sleep efficiency', axis=1)
y = sleep_update2['Sleep efficiency']
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""# *Making Predictions*
"""

# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred_test = regressor.predict(X_test)
#y_pred_train = regressor.predict(X_train)

# Print the coefficients and intercept
print('Coefficients:', regressor.coef_)
print('Intercept:', regressor.intercept_)

# Evaluate the model, obtain the mse and R2
from sklearn.metrics import mean_squared_error, r2_score
mse_linear = mean_squared_error(y_test, y_pred_test)
r2_linear = r2_score(y_test, y_pred_test)
print("Mean squared error:",  mean_squared_error(y_test, y_pred_test))
print("R-squared:", r2_score(y_test, y_pred_test))

# Fit the linear regression
coeff = np.polyfit(X_train['Age'].values, y_train.values, 1)
m, b = coeff

# Create a scatter plot
plt.scatter(X_train['Age'], y_train)

# Plot the regression line
plt.plot(X_train['Age'], m*X_train['Age'] + b, color='red')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('y')
plt.title('Linear Regression')

# Show the plot
plt.show()

"""# ***Multivariate Polynomial Regression*** - comparison model
"""

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Create a PolynomialFeatures obj with degree 2
poly = PolynomialFeatures(degree = 2)
# Transform the training data
X_train_poly = poly.fit_transform(X_train)
# Transforms test data using same transformation learned from training data
X_test_poly = poly.transform(X_test)

# Create and fit LinearRegression model using the transformed training data
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Create a PolynomialFeatures obj with degree 2
poly = PolynomialFeatures(degree = 2)
# Transform the training data
X_train_poly = poly.fit_transform(X_train)
# Transforms test data using same transformation learned from training data
X_test_poly = poly.transform(X_test)

# Create and fit LinearRegression model using the transformed training data
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predict the model
predict = poly_reg.predict(X_test_poly)
print(len(predict))
print(predict)

# Get the coefficients and intercept
coefficients = poly_reg.coef_
intercept = poly_reg.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

from sklearn.metrics import mean_squared_error, r2_score
# Obtain the mse and r2 for the polynomial regression model
mse_poly = mean_squared_error(y_test, predict)
r2_poly = r2_score(y_test, predict)
print("Mean squared error:",  mean_squared_error(y_test, predict))
print("R-squared:", r2_score(y_test, predict))

## Comparison of both models
# mse
print("Linear Regression MSE:", mse_linear)
print("Polynomial Regression MSE:", mse_poly)
# r2
print("Linear Regression R2:", r2_linear)
print("Polynomial Regression R2:", r2_poly)

# Choose the model with the lowest MSE
if mse_linear < mse_poly:
    print("Linear Regression performs better")
else:
    print("Polynomial Regression performs better")

# Choose the model with the highest R2
if r2_linear > r2_poly:
    print("Linear Regression performs better")
else:
    print("Polynomial Regression performs better")
