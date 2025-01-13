
# Spam Detection Using NLTK and FFNN 

We are using a email classfication dataset with 3002 Column and 5112 rows of data.
We are using Natrual Language Toolkit and FFNN to train a model to detect is the Email is SPAM or not. 

Mission of this project is to learn and train a model to help people to detect the SPAM Mail. So they can safe out from virus and malware and other spam thing.


## Information Section

[Download Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

## Documentation

In Both Model We have use the 95% of the data from dataset. and 5% of data for testing.

## FFNN

* Importing Packages
```python
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

```

## Approach 1

* NLTK Stopword Setup and Filtering
```python
nltk.download('stopwords')

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
```

* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('./emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]
```

* Traning the Model and Using 100 Epochs and Using 30 batch Size.
```python
main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.2, random_state=56)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)


model = Sequential([
    Dense(128, activation='relu', input_shape=(Main_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=100, batch_size=30,
                   validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)
```

![Epoch](/screenshot/ep-1.png)


![Confisuon Martrix, F1](/screenshot/ffnn-cof1.png)

![Validation Map](/screenshot/ffnn-1.png)


## Approach 2 Using LSTM

* NLTK Stopword Setup and Filtering
```python
nltk.download('stopwords')

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
```

* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('./emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]
```

* Traning the Model and Using 100 Epochs and Using 30 batch Size.
```python
main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.2, random_state=56)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)


model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),  # Embedding layer
    Bidirectional(LSTM(64, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=100, batch_size=30,
                   validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)
```

![Epoch](/screenshot/ep-2.png)

![Confisuon Martrix, F1](/screenshot/ffnn-cof2.png)

![Validation Map](/screenshot/ffnn-2.png)
