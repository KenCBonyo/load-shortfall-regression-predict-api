"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('C:/Users/mufre/OneDrive/Desktop/DATABASES/API/load-shortfall-regression-predict-api/utils/data/df_train.csv')

y_train = train[['load_shortfall_3h']]
#Replace the null values in Valencia_pressure with Madrid_pressure values on the same row.
columns =[col for col in train.columns if col!='load_shortfall_3h']
train_copy_df=train[columns]

train_copy_df.loc[train_copy_df['Valencia_pressure'].isna(),'Valencia_pressure'] = \
     train_copy_df.loc[train_copy_df['Valencia_pressure'].isna(), 'Madrid_pressure']
# Extracting year from time column 
train_copy_df['time'] = pd.to_datetime(train_copy_df.time)
train_copy_df['year'] = train_copy_df[['time']].applymap(lambda dt:dt.year
 if not pd.isnull(dt.year) else 0)
train_copy_df['month'] = train_copy_df[['time']].applymap(lambda dt:dt.month
 if not pd.isnull(dt.month) else 0)
train_copy_df['Day'] = train_copy_df[['time']].applymap(lambda dt:dt.day
 if not pd.isnull(dt.day_name()) else 0)
train_copy_df['Hours'] = train_copy_df[['time']].applymap(lambda dt:dt.hour
 if not pd.isnull(dt.day_name()) else 0)
     
train_copy_df.loc[train_copy_df['month'].isin([1,2,3]),['winter','spring','summer','autumn']] = [1,0,0,0]
train_copy_df.loc[train_copy_df['month'].isin([4,5,6]),['winter','spring','summer','autumn']] = [0,1,0,0]
train_copy_df.loc[train_copy_df['month'].isin([7,8,9]),['winter','spring','summer','autumn']] = [0,0,1,0]
train_copy_df.loc[train_copy_df['month'].isin([10,11,12]),['winter','spring','summer','autumn']] = [0,0,0,1]

train_copy_df = train_copy_df.astype(
    {
        'winter': int, 'summer': int, 'spring': int, 'autumn': int
    }
)
   
   
dummies_df = pd.get_dummies(train_copy_df[['Valencia_wind_deg','Seville_pressure']], drop_first = True)
train_copy_df = pd.concat([train_copy_df, dummies_df], axis='columns')
train_copy_df = train_copy_df.drop(['Valencia_wind_deg', 'Seville_pressure' ], axis='columns')

column_titles = [col for col in train_copy_df.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
train_copy_df = train_copy_df.reindex(columns = column_titles)


     
     
     
     
X_train = train_copy_df[['Madrid_wind_speed', 'Bilbao_rain_1h', 'Valencia_wind_speed',
     'Seville_humidity','Madrid_humidity', 'Bilbao_clouds_all',
     'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg',
     'Barcelona_wind_speed', 'Barcelona_wind_deg' ,'Madrid_clouds_all',
     'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h',
     'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
     'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure',
     'Seville_weather_id', 'Valencia_pressure', 'Seville_temp_max',
     'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
     'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
     'Barcelona_temp_max', 'Madrid_temp_max' ,'Barcelona_temp', 'Bilbao_temp_min',
     'Bilbao_temp', 'Barcelona_temp_min','Bilbao_temp_max', 'Seville_temp_min',
     'Madrid_temp' ,'Madrid_temp_min', 'year' ,'month', 'Hours', 'winter' ,'spring',
     'summer', 'autumn', 'Valencia_wind_deg_level_10',
     'Valencia_wind_deg_level_2', 'Valencia_wind_deg_level_3',
     'Valencia_wind_deg_level_4', 'Valencia_wind_deg_level_5',
     'Valencia_wind_deg_level_8', 'Valencia_wind_deg_level_9',
     'Seville_pressure_sp12', 'Seville_pressure_sp24', 'Seville_pressure_sp25',
     'Seville_pressure_sp4']]

# Fit model
cls=RandomForestRegressor(n_estimators= 250,
                                    max_depth= 10,
                                    random_state= 42)
print ("Training Model...")
cls.fit(X_train, y_train)

# Pickle model for use within our API
save_path = 'C:/Users/mufre/OneDrive/Desktop/DATABASES/API/load-shortfall-regression-predict-api/assets/trained-models/es5_rfr_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(cls, open(save_path,'wb'))
