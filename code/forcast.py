from turtle import st
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pylab as plt
import argparse
import os


class Forcast:
    def read_data(data_path):
        data = pd.read_csv(data_path,  sep=',', index_col="Date")
        data.index = pd.to_datetime(data.index)
        return data

    def create_features(data):
        # Creates time series features from datetime index
        # convert data column to datetime which will make it easy to dealing with time data
        data['Date_'] = data.index
        data['Date_'] = pd.to_datetime(data['Date_'])
        data['Date_'] = pd.to_datetime(data['Date_'],format='%m/%d/%Y')
        data['year']=data['Date_'].dt.year 
        data['month']=data['Date_'].dt.month 
        data['day']=data['Date_'].dt.day
        data['dayofweek']=data['Date_'].dt.dayofweek  
        data.drop("Date_",axis=1,inplace=True)
        
        # adding more features using by shift the columns
        data['lag_1'] = data['Price'].shift(1)
        data['lag_2'] = data['Price'].shift(2)
        data['lag_3'] = data['Price'].shift(3)
        data['lag_4'] = data['Price'].shift(4)
        data['lag_5'] = data['Price'].shift(5)
        data['lag_6'] = data['Price'].shift(6)
        data['lag_7'] = data['Price'].shift(7)

        #adding multiple features using rolling window
        data['rolling_mean_2'] = data['Price'].rolling(window=2).mean()
        data['rolling_mean_3'] = data['Price'].rolling(window=3).mean()
        data['rolling_mean_4'] = data['Price'].rolling(window=4).mean()
        data['rolling_mean_5'] = data['Price'].rolling(window=5).mean()
        data['rolling_mean_6'] = data['Price'].rolling(window=6).mean()
        data['rolling_mean_7'] = data['Price'].rolling(window=7).mean()

        #adding addional features using expanding window
        data['expanding_mean'] = data['Price'].expanding(2).mean()
        data['expanding_mean'] = data['Price'].expanding(3).mean()
        data['expanding_mean'] = data['Price'].expanding(4).mean()
        data['expanding_mean'] = data['Price'].expanding(5).mean()
        data['expanding_mean'] = data['Price'].expanding(6).mean()
        data['expanding_mean'] = data['Price'].expanding(7).mean()
        
        label = "Price"
        features = [col for col in data.columns if col!= label]
        
        x = data[features]
        y = data[label]  

        return x,y

    def split_data(data, _train_ratio=0.8):
        # split the data to train and test first 80% for train. 20% for test
        split = int(len(data) * _train_ratio)
        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()
        return test, train
        
    def create_model():
        model = xgb.XGBRegressor()
        return model 

    def fit_model(model, X_train, y_train):
        model.fit(X_train, y_train, verbose=False)
        return model

    def save_model():
        model_path = os.path.join('models', 'model.json')
        model.save_model(model_path)

    def evaluate_model(model, X_test, train, test):
        # plot the features importance
        plot_importance(model, height=0.8)
        # predict the stock price for the test set
        test['Prediction'] = model.predict(X_test)
        all_data = pd.concat([test, train], sort=False)
        # plot the acutal and predicte stock price
        all_data[['Price','Prediction']].plot(figsize=(15, 5), style=['-','-'])
        #print the MSE, MAE
        mse = mean_squared_error(y_true=test['Price'],
                    y_pred=test['Prediction'])
        mae = mean_absolute_error(y_true=test['Price'],
                    y_pred=test['Prediction'])
        mape = mean_absolute_percentage_error(y_true=test['Price'],
                    y_pred=test['Prediction'])
        print("MSE : ",mse)
        print("MAE : ",mae)
        print("MAPE: ",mape)
    
    
    
if __name__ == "__main__":
    
    default_data_path = os.path.join('data', 'prices.txt')
    parser = argparse.ArgumentParser(description='Time-series forcasting.')
    parser.add_argument('--data', help='Path of data.', default=default_data_path)
    parser.add_argument('--ratio', help='training ratio.', default=0.8)
    args = parser.parse_args()
    # read the data path from the command-line
    data_path = args.data
    # read the training ratio 
    ratio = args.ratio
    
    
    data = Forcast.read_data(data_path)
    test, train = Forcast.split_data(data, ratio)
    X_train, y_train = Forcast.create_features(train)
    X_test, y_test = Forcast.create_features(test)
    model = Forcast.create_model()
    model = Forcast.fit_model(model, X_train, y_train)
    Forcast.save_model()
    Forcast.evaluate_model(model, X_test, train, test)


    
    