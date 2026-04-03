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