import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing

print(tf.__version__)

BUCKET = 'gs://vertex-ai-lab-487403-heet'

# Load California Housing dataset
housing = fetch_california_housing()
dataset = pd.DataFrame(housing.data, columns=housing.feature_names)
dataset['Price'] = housing.target

print("Dataset shape:", dataset.shape)
print(dataset.head())

# Split into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Get stats for normalization (remove Price from stats)
train_stats = train_dataset.describe()
train_stats.pop('Price')
train_stats = train_stats.transpose()

# Split features from labels
train_labels = train_dataset.pop('Price')
test_labels = test_dataset.pop('Price')

# Normalize
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()

# Train with early stopping
EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels,
                          epochs=EPOCHS, validation_split=0.2,
                          callbacks=[early_stop])

# Evaluate
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print(f"Test MAE: {mae}, Test MSE: {mse}")

# Export model to GCS
model.save(BUCKET + '/california_housing/model')
print("Model saved to", BUCKET + '/california_housing/model')
