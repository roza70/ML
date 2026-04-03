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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Identify and remove classes with only 1 member
# This counts how many times each label appears
label_counts = pd.Series(y).value_counts()
to_keep = label_counts[label_counts >= 2].index

# 2. Filter X and y to only include those classes
mask = np.isin(y, to_keep)
X_filtered = X[mask]
y_filtered = y[mask]

# 3. Now perform the Split safely
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, 
    y_filtered, 
    test_size=0.20, 
    random_state=42,
    stratify=y_filtered
    )

print("Train-Test Split done")
print(f"Dropped {len(y) - len(y_filtered)} rows with insufficient class members.")
print(f"Training Rows: {X_train.shape[0]}")
print(f"Testing Rows:  {X_test.shape[0]}")
Train-Test Split done
Dropped 1 rows with insufficient class members.
Training Rows: 89756
Testing Rows:  22439
