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
 Re-mapping labels to start at 0 (Required for Keras/TensorFlow)
# This turns label 1 -> 0 and label 2 -> 1
y_train = y_train - 1
y_test = y_test - 1

print("Labels re-mapped to [0, 1]")
print(f"New unique labels in training: {np.unique(y_train)}")
Labels re-mapped to [0, 1]
New unique labels in training: [0 1]


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. RANDOM FOREST (Fast and Accurate for 471 features)
print("Training Random Forest on 112,196 rows...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"\n[RF] Accuracy: {rf_acc:.4f}")

# 2. CNN (Deep Learning)
print("\nPreparing CNN (Reshaping data)...")
X_train_cnn = np.expand_dims(X_train.values, axis=-1)
X_test_cnn = np.expand_dims(X_test.values, axis=-1)
num_classes = len(np.unique(y_train))

# Simplified CNN for high-dimensional feature vectors
cnn = models.Sequential([
    layers.Input(shape=(X_train_cnn.shape[1], 1)),
    layers.Conv1D(16, 3, activation='relu'),
    layers.GlobalAveragePooling1D(), # Better for large feature sets
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training CNN (5 Epochs)...")
cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

cnn_acc = cnn.evaluate(X_test_cnn, y_test, verbose=0)[1]


print("\n" + "="*40)
print("  FINAL RESULTS")
print("="*40)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
print(f"CNN Accuracy:           {cnn_acc*100:.2f}%")
print("\n" + "="*40)


# Optional: Print detailed report for RF
print("\nDetailed Classification Report (Random Forest):")
print(classification_report(y_test, rf_preds))
