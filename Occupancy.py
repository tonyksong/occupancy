# ### Problem Statement:
# Detect future occupancy of a room given parameters light, temperature, humidity and CO2.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz,plot_tree
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [10.0, 6.0]


df_train = pd.read_csv('train_data.txt')
df_val = pd.read_csv('validation_data.txt')

# Explore data
df_train.head()
df_train.tail()

# Data seems to be ordered by date and time and appropriate for time series analysis. Time intervals seem to be about every minute. Dates are from 2015-02-04 to 2015-02-10 (6 days).

df_train.info()

# There are no null values in the training data. May need to convert date to datetime object. Occupancy is the only int type.

df_train.describe()


# Occupancy seems to be binary: either 0 or 1. Assume 1 means occupied, and 0 means unoccupied.

# Convert date column to datetime object
df_train['date'] = pd.to_datetime(df_train['date'])

# Verify changed datatype for date
df_train.dtypes


# Date values are either at :00 or :59 seconds. Round all to nearest minute.

# Set date to index and round to nearest minute for time series analysis
df_train = df_train.set_index('date')
df_train.index = df_train.index.round('min')
df_train.head()

pd.plotting.register_matplotlib_converters()
# Plot all columns vs. date
df_train.plot(subplots=True, figsize=(15,10))
plt.show()


# There appears to be some correlation between the variables, which can be seen from when the peaks occur. The occupancy seems less correlated to Humidity than it does with Light and CO2. Light seems to be the most important factor, since there are dips in the occupancy that coincide with dips in Light. There seems to be periodicity in the occupancy, where the occupancy can be 1 during "normal working hours" (i.e., 9am-6pm). However, there was no occupancy on 2 consecutive dates, 2015-02-07 and 2015-02-08. Perhaps those dates were the weekend, which indicates conditional seasonality. For those 2 dates, the CO2 levels were also very low with no peaks. Since the date range is small, assume no overall underlying trends in the variables. There also seems to be a high outlier in the Light data, with the value of 1546.3 within the day of 2015-02-07.

# Check for Seasonality in parameters

# Perform seasonality decomposion for Temperature, period = 1 day
seas_d = seasonal_decompose(df_train['Temperature'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()

# Perform seasonality decomposion for Light, period = 1 day
seas_d = seasonal_decompose(df_train['Light'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()

# Perform seasonality decomposion for CO2, period = 1 day
seas_d = seasonal_decompose(df_train['CO2'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()

# Perform seasonality decomposion for HumidityRatio, period = 1 day
seas_d = seasonal_decompose(df_train['HumidityRatio'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()


# Perform Granger Causality Tests

# Examine Temperature causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Temperature']], maxlag=4))


# For a lag <= 2, the Temperature seems to cause Occupancy based on the Granger Causality Test p-value < 0.05.

# Examine Humidity causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Humidity']], maxlag=4))


# The null hypothesis fails to be rejected at the p-value of 0.05, therefore Humidity does not cause Occupancy.

# Examine Light causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Light']], maxlag=4))


# Light seems to have a strong effect on Occupancy, since p-value is 0.000.

# Examine CO2 causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'CO2']], maxlag=4))


# CO2 seems to have a strong effect on Occupancy, since p-value is < 0.05 for max_lag <= 4.

# Examine HumidityRatio causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'HumidityRatio']], maxlag=4))


# The null hypothesis fails to be rejected at the p-value of 0.05, therefore HumidityRatio does not cause Occupancy.

# Check each parameter for stationarity using ADF Test

# Check Temperature for stationarity
result = adfuller(df_train['Temperature'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and Temperature is NOT stationary.

# Check Humidity for stationarity
result = adfuller(df_train['Humidity'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and Humidity is NOT stationary.

# Check Light for stationarity
result = adfuller(df_train['Light'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# The p-value is below the rule-of-thumb threshold of 0.05, indicating that there is no unit root, and Light IS stationary.

# Check CO2 for stationarity
result = adfuller(df_train['CO2'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# The p-value is below the rule-of-thumb threshold of 0.05, indicating that there is no unit root, and CO2 IS stationary.

# Check HumidityRatio for stationarity
result = adfuller(df_train['HumidityRatio'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and HumidityRatio is NOT stationary.

# Examine hours for Occupancy=1
df_train[df_train.index.day == 5]['Occupancy'].plot()
plt.show()

df_train[df_train.index.day == 6]['Occupancy'].plot()
plt.show()

df_train[df_train.index.day == 9]['Occupancy'].plot()
plt.show()


# The plots confirm that the hours for occupancy tend to be roughly between 9am-6pm.

# Since the day of the week and time of the day seem to be important factors in predicting occupancy and trends in the other variables, create new columns for "WorkHours" and "Weekday"

# Separate X and y
X_df_train = df_train.iloc[:,:5]
y_df_train = df_train['Occupancy']

# Add binary column for Weekday
X_df_train['Weekday'] = np.where((X_df_train.index.day == 7) | (X_df_train.index.day == 8), 0, 1)

# Add binary column for WorkHours
X_df_train['WorkHours'] = np.where((X_df_train.index.hour >= 8) & (X_df_train.index.day <= 6), 1, 0)

# Examine outlier for Light
X_df_train[X_df_train.index.day == 7]['Light'].plot(grid=True)
plt.show()


# This is most likely an error, so we'll assume that the data for that spike can be removed and imputed.

# Find time when spike occurred
X_df_train[(X_df_train.index.day == 7) & (X_df_train.index.hour == 9) & (X_df_train['Light'] > 400)]['Light']


# The data for Light was unusually high for 4 data points (09:40:59 through 09:43:59). To improve training the model, we'll change these values by imputing using simple linear regression.

# Replace high values with NaN
X_df_train.at['2015-02-07 09:41:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:42:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:43:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:44:00', 'Light'] = np.nan


# Interpolate NaN values using simple linear regression
X_df_train['Light'].interpolate(method='linear', inplace=True)

# Verify values were imputed
X_df_train[X_df_train.index.day == 7]['Light'].plot(grid=True)
plt.show()


# CO2 also had some unusually high spikes near the end of the day on 2015-02-09. Let's zoom in and examine it.

# Examine spikes for CO2
df_train[df_train.index.day == 9]['CO2'].plot(grid=True)
plt.show()


# However, the spike for CO2 is still in the range of the data, so we'll keep it and not bother smoothing it for now.

# Examine correlation matrix for predicting variables
X_df_train.corr()


# The correlation matrix shows that Humidity and HumidityRatio are highly correlated, so one of these may potentially be removed from the model to reduce multicollinearity. CO2 and HumidityRatio seem somewhat correlated, too. Therefore, let us drop HumidityRatio from the model.

X_df_train = X_df_train.drop(columns = 'HumidityRatio')

# Examine correlation matrix after dropping HumidityRatio
X_df_train.corr()


# Inspect df_val:
df_val.info()


# Again, no null values.
df_val.describe()

# Convert date column to datetime object and set as index, round to nearest minute
df_val['date'] =  pd.to_datetime(df_val['date'])
df_val = df_val.set_index('date')
df_val.index = df_val.index.round('min')

# Plot df_val to visualize
df_val.plot(subplots=True, figsize=(15,10))
plt.show()


# It can be seen that this is a continuation of the time series data from the training set. Now, the dates 2015-02-11 to 2015-02-19 are given. The same seasonality pattern of no occupancy for 2 consecutive days (2015-02-14 and 2015-02-15) can be observed, which indicates conditional seasonality. Add columns for Weekday or Weekend and Hour to take into account this seasonality for predicting Occupancy. Also, note that the time range is quite short here, and none of the variables lead us to predict that there is a trend component to this time series. Therefore, assume there is no trend component.

# Drop HumidityRatio from df_val
df_val = df_val.drop(columns = 'HumidityRatio')

# Separate X and y
X_df_val = df_val.iloc[:,:4]
y_df_val = df_val['Occupancy']

# Add columns for Weekday and WorkHours
X_df_val['Weekday'] = np.where((X_df_val.index.day == 14) | (X_df_val.index.day == 15), 0, 1)
X_df_val['WorkHours'] = np.where((X_df_val.index.hour >= 8) & (X_df_val.index.hour <= 6), 1, 0)


# Split the validation set into validation set for comparing models and test set for assessing model accuracy. Use dates from 2015-02-11 to 2015-02-14 for validation set, and 2015-02-15 to 2015-02-18 for test set. This gives a good balance by including a date in each set with a "weekend" day, and makes the test set the latest set in time.

X_df_test = X_df_val[df_val.index.day >= 15]
y_df_test = y_df_val[df_val.index.day >= 15]
X_df_val = X_df_val[df_val.index.day <= 14]
y_df_val = y_df_val[df_val.index.day <= 14]

# Normalize appropriate columns for building Classification Models
cols_to_norm = ['Temperature', 'Humidity', 'Light', 'CO2']
X_df_train[cols_to_norm] = X_df_train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X_df_val[cols_to_norm] = X_df_val[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_df_test[cols_to_norm] = X_df_test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


X_train = X_df_train
y_train = y_df_train
X_test = X_df_val
y_test = y_df_val

# Implement Logistic Regression
logreg = LogisticRegression(solver='lbfgs', max_iter=200)
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = logreg.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Implement the Gaussian Naive Bayes algorithm for classification
gnb = GaussianNB()
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Implement KNN
neigh = KNeighborsClassifier(n_neighbors=3) #Best: n_neighbors=3
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = neigh.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using svm
SVM = svm.SVC(C=1, kernel = 'rbf').fit(X_train, y_train) #Best: C=1, kernel = 'rbf'
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using simple neural network
NN = MLPClassifier(solver = 'adam', learning_rate_init = 0.001, max_iter = 250, hidden_layer_sizes=(5, 2), random_state=99).fit(X_train, y_train)
y_pred = NN.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using CART
dct = DecisionTreeClassifier(max_depth=5, random_state=99)
dct.fit(X_train, y_train)
y_pred = dct.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Plot decision tree to view feature importances
plt.figure(figsize=(25,10)) #Set figure size for legibility
plot_tree(dct, filled=True)
plt.show()


# The Decision Tree plot shows that X[2] (Light) is the most important feature, followed by X[3] (CO2) and X[0] (Temperature).

# Build classifier using Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=99)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Calculate feature importances
importances = rf.feature_importances_
print(importances)

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
print(indices)


# The rank and weights of the features from the Random Forest classifier confirm that X[2] (Light) is the most important feature.

# #### SVM model with C=1 and kernel='rbf' performed the best on the validation set. Assess the true accuracy of the model using the test set.


X_test = X_df_test
y_test = y_df_test

# Build classifier using svm
SVM = svm.SVC(C=1, kernel = 'rbf').fit(X_train, y_train) #Best: C=1, kernel = 'rbf'
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))


# The accuracy on the test set is 0.996107, which is very high!

# Plot confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(SVM, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# ### The model above only predicts based on the variables Temperature, Humidity, CO2, Weekday, and WorkHours. It assumes that the information on the variables is known. For further exploration, subsets of the variables could be used to train more models. For a more difficult problem, forecast the variables and make predictions from the forecast, and compare the predictions with the actual results.

# Fit time series forecasting models with Facebook Prophet

# The range of date for the df_train set is 2015-02-04 17:51:00 (Wed) through 2015-02-10 09:33:00 (Tues).<br>
# The range of date for the df_val set is 2015-02-11 14:48:00 (Wed) through 2015-02-14 23:59:00 (Sat).<br>
# The range of date for the df_test set is 2015-02-15 through (Sun) 2015-02-18 09:19:00 (Wed).<br>
# Fit the predicted time series for the individual parameters to align with the day of week and time on the df_val set.

# Prophet requires columns ds (Date) and y (value)
X_df_train_forecast = df_train[['Temperature', 'Humidity', 'Light', 'CO2']]
X_df_train_forecast['ds'] = df_train.index
X_df_train_forecast['y'] = df_train['Temperature']

X_df_train_forecast

def is_weekday(ds):
    date = pd.to_datetime(ds)
    return (date.day != 7 and date.day != 8 and date.day != 14 and date.day != 15)

X_df_train_forecast['weekday'] = X_df_train_forecast['ds'].apply(is_weekday)
X_df_train_forecast['weekend'] = ~X_df_train_forecast['ds'].apply(is_weekday)

# Model seasonality to make forecasts for Temperature
m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
m.add_seasonality(name='daily_weekday', period=1, fourier_order=3, condition_name='weekday')
m.add_seasonality(name='daily_weekend', period=1, fourier_order=3, condition_name='weekend')
m.fit(X_df_train_forecast)
# Make a future dataframe up to 2015-02-18 09:19:00
temp_forecast = m.make_future_dataframe(periods=60*24*8, freq='T')
temp_forecast['weekday'] = temp_forecast['ds'].apply(is_weekday)
temp_forecast['weekend'] = ~temp_forecast['ds'].apply(is_weekday)
# Make predictions
forecast = m.predict(temp_forecast)

m.plot(forecast, xlabel = 'Date', ylabel = 'Temperature')
plt.title('Temperature');

X_df_train_forecast['TempForecast'] = forecast['yhat']


temp_fc = forecast[['ds', 'yhat']].rename(columns={"ds": "date", "yhat": "TempForecast"}).set_index('date')

df_val = df_val.join(temp_fc, on='date', how='inner')

df_val.plot(y=["Temperature", "TempForecast"], figsize=(15,4))


# Model seasonality to make forecasts for Humidity
m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
m.add_seasonality(name='daily_weekday', period=1, fourier_order=3, condition_name='weekday')
m.add_seasonality(name='daily_weekend', period=1, fourier_order=3, condition_name='weekend')
X_df_train_forecast['y'] = df_train['Humidity']
m.fit(X_df_train_forecast)
# Make a future dataframe up to 2015-02-18 09:19:00
hum_forecast = m.make_future_dataframe(periods=60*24*8, freq='T')
hum_forecast['weekday'] = hum_forecast['ds'].apply(is_weekday)
hum_forecast['weekend'] = ~hum_forecast['ds'].apply(is_weekday)
# Make predictions
forecast = m.predict(hum_forecast)

X_df_train_forecast['HumForecast'] = forecast['yhat']
hum_fc = forecast[['ds', 'yhat']].rename(columns={"ds": "date", "yhat": "HumForecast"}).set_index('date')
df_val = df_val.join(hum_fc, on='date', how='inner')
df_val.plot(y=["Humidity", "HumForecast"], figsize=(15,4))


# Model seasonality to make forecasts for Light
m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
m.add_seasonality(name='daily_weekday', period=1, fourier_order=3, condition_name='weekday')
m.add_seasonality(name='daily_weekend', period=1, fourier_order=3, condition_name='weekend')
X_df_train_forecast['y'] = df_train['Light']
m.fit(X_df_train_forecast)
# Make a future dataframe up to 2015-02-18 09:19:00
light_forecast = m.make_future_dataframe(periods=60*24*8, freq='T')
light_forecast['weekday'] = light_forecast['ds'].apply(is_weekday)
light_forecast['weekend'] = ~light_forecast['ds'].apply(is_weekday)
# Make predictions
forecast = m.predict(light_forecast)
X_df_train_forecast['LightForecast'] = forecast['yhat']
light_fc = forecast[['ds', 'yhat']].rename(columns={"ds": "date", "yhat": "LightForecast"}).set_index('date')
df_val = df_val.join(light_fc, on='date', how='inner')
# Plot Light vs. Forecasted Light
df_val.plot(y=["Light", "LightForecast"], figsize=(15,4))

# Model seasonality to make forecasts for CO2
m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
m.add_seasonality(name='daily_weekday', period=1, fourier_order=3, condition_name='weekday')
m.add_seasonality(name='daily_weekend', period=1, fourier_order=3, condition_name='weekend')
X_df_train_forecast['y'] = df_train['CO2']
m.fit(X_df_train_forecast)
# Make a future dataframe up to 2015-02-18 09:19:00
co2_forecast = m.make_future_dataframe(periods=60*24*8, freq='T')
co2_forecast['weekday'] = co2_forecast['ds'].apply(is_weekday)
co2_forecast['weekend'] = ~co2_forecast['ds'].apply(is_weekday)
# Make predictions
forecast = m.predict(co2_forecast)
X_df_train_forecast['CO2Forecast'] = forecast['yhat']
co2_fc = forecast[['ds', 'yhat']].rename(columns={"ds": "date", "yhat": "CO2Forecast"}).set_index('date')
df_val = df_val.join(co2_fc, on='date', how='inner')
df_val.plot(y=["CO2", "CO2Forecast"], figsize=(15,4))


# Predict Occupancy with forecasted parameters

# Separate X and y
X_df_val = df_val[['TempForecast', 'HumForecast', 'LightForecast', 'CO2Forecast']]
y_df_val = df_val['Occupancy']

# Add columns for Weekday and WorkHours
X_df_val['Weekday'] = np.where((X_df_val.index.day == 14) | (X_df_val.index.day == 15), 0, 1)
X_df_val['WorkHours'] = np.where((X_df_val.index.hour >= 8) & (X_df_val.index.hour <= 6), 1, 0)

# Split into validation and test sets
X_df_test = X_df_val[df_val.index.day >= 15]
y_df_test = y_df_val[df_val.index.day >= 15]
X_df_val = X_df_val[df_val.index.day <= 14]
y_df_val = y_df_val[df_val.index.day <= 14]

# Normalize appropriate columns for building Classification Models
cols_to_norm = ['TempForecast', 'HumForecast', 'LightForecast', 'CO2Forecast']
X_df_val[cols_to_norm] = X_df_val[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_df_test[cols_to_norm] = X_df_test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Rename variables for easier implementation
X_train = X_df_train
y_train = y_df_train
X_test = X_df_val
y_test = y_df_val

# Implement Logistic Regression
logreg = LogisticRegression(solver='lbfgs', max_iter=100)
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = logreg.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Implement the Gaussian Naive Bayes algorithm for classification
gnb = GaussianNB()
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Implement KNN
neigh = KNeighborsClassifier(n_neighbors=3) #Best: n_neighbors=3
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = neigh.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using SVM
SVM = svm.SVC(C=.01, kernel = 'rbf').fit(X_train, y_train) #Best: C=.01, kernel = 'rbf'
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using simple neural network
NN = MLPClassifier(solver = 'adam', learning_rate_init = 0.01, max_iter = 150, hidden_layer_sizes=(5, 2), random_state=99).fit(X_train, y_train)
y_pred = NN.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using CART
dct = DecisionTreeClassifier(max_depth=4, random_state=99)
dct.fit(X_train, y_train)
y_pred = dct.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# Build classifier using Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=1, random_state=99)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))


# SVM with C=0.01 and kernel='rbf' performed the best in the validation set. Test its true accuracy on the test set.

X_test = X_df_test
y_test = y_df_test
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))

# The accuracy on the test set was 0.897336. Not bad. The accuracy would be improved if the variables were forecasted better. Now plot the confusion matrix.

# Plot confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(SVM, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# ### The final model chosen to predict Occupancy when the variables Temperature, Humidity, Light, and CO2 were forecasted from the training set was a Support Vector Model with margin C=0.01 and kernel='rbf'. This model performed the best in the validation set and resulted in a 89.7% accuracy in the test set. The confusion matrix shows that out of 4880 total test data points (minutes), 3282 points were correctly identified as unoccupied (true negatives), 1097 points were correctly identified as occupied (true positives), 501 points were incorrectly identified as occupied when they were unoccupied (false positives), and 0 points were incorrectly identified as unoccupied when they were occupied (false negatives). For the future, the model could be improved by tuning the trend component for forecasting the variables, and building and testing models on subsets of the parameters.


