import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

path = 'RainForecast_LinearRegression/data/raw/weather.csv'
df = pd.read_csv(path)

#Altering and adding a few columns
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.drop(columns=['Year', 'Month', 'Day'])
cols = ['Date'] + [col for col in df.columns if col != 'Date']
df = df[cols]
df.insert(1, 'Quarter', df['Date'].dt.quarter)
idx_rel_humidity = df.columns.get_loc('Relative Humidity')
df.insert(idx_rel_humidity + 1, 'Humidity Difference', df['Specific Humidity'] - df['Relative Humidity'])
temp_diff = df['Temperature'].diff()
mean_temp_change = temp_diff.mean()
df['Temp Change'] = temp_diff.fillna(mean_temp_change)
idx_temperature = df.columns.get_loc('Temperature')
temp_change_col = df.pop('Temp Change')
df.insert(idx_temperature + 1, 'Temp Change', temp_change_col)

#Standardization
numeric_cols = [
    'Specific Humidity', 'Relative Humidity', 'Humidity Difference', 
    'Temperature', 'Temp Change', 'Precipitation'
]
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df_standardized.info()

out_path = 'RainForecast_LinearRegression/data/processed/weather.csv'
df_standardized.to_csv(out_path, index=False)

print('Created processed weather.csv successfully!')