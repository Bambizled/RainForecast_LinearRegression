import pandas as pd
import numpy as np

path = 'RainForecast_LinearRegression/data/raw/weather.csv'
df = pd.read_csv(path)
df.info()
df.describe()
df.head()
