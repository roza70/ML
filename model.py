import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

import os

path = "/kaggle/input/datasets/agungpambudi/secure-intrusion-detection-ddos-attacks-profiling"
print(os.listdir(path))
['data-dictionary.csv', 'ddos-attacks-profiling']
import pandas as pd
path = "/kaggle/input/datasets/agungpambudi/secure-intrusion-detection-ddos-attacks-profiling/data-dictionary.csv"
df = pd.read_csv(path, sep='|') 
df.columns = df.columns.str.strip()
print("Data Shape:", df.shape)
Data Shape: (7, 2)

df.isnull().sum()
Feature Name    0
Description     0
dtype: int64

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Fill missing values (just in case there are any after the correct load)
df = df.fillna(df.mode().iloc[0])

# 2. Label Encoding
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 3. Normalization
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Data processing complete!")
Data processing complete!