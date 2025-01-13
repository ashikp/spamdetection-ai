import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

import nltk
from nltk.corpus import stopwords


# Download the stopwords if you haven't already
nltk.download('stopwords')

# Get the list of English stopwords
stop_words = stopwords.words('english')

print(stop_words)

dataset = pd.read_csv('/content/drive/MyDrive/Data Extract/nltk_project.csv')
df_main = dataset.drop(columns=["Email No.", "Prediction"])

print(df_main.sample(20))

type(stop_words)

filter = [item for item in df_main.columns if item not in stop_words]

print(filter)
print(len(df_columns))
print(len(filter))

dataset = pd.read_csv('/content/drive/MyDrive/Data Extract/nltk_project.csv')
df = pd.DataFrame(dataset, columns=filter)
print(len(df.columns))

"""## Approach 1"""

Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]

main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.05, random_state=101)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(Main_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=20, batch_size=32, validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)

conf_matrix = confusion_matrix(main_predication_test, main_trained)
f1 = f1_score(main_predication_test, main_trained)
class_report = classification_report(main_predication_test, main_trained)

print("Confusion Matrix:")
print(conf_matrix)

print("\nF1 Score:", f1)

print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]

main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.2, random_state=56)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)

# Model Architecture
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),  # Embedding layer
    Bidirectional(LSTM(64, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(Main_train, main_predication_train, epochs=10, batch_size=32, validation_data=(Main_test, main_predication_test), verbose=1)
main_trained = (model.predict(Main_test) > 0.5).astype(int)

conf_matrix = confusion_matrix(main_predication_test, main_trained)
f1 = f1_score(main_predication_test, main_trained)
class_report = classification_report(main_predication_test, main_trained)

print("Confusion Matrix:")
print(conf_matrix)

print("\nF1 Score:", f1)

print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

