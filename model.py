from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
import os

ex = Experiment("s464906_experiment")

mongo_url = os.getenv("MONGO_URL", "mongodb://admin:IUM_2021@tzietkiewicz.vm.wmi.amu.edu.pl:27017")
ex.observers.append(MongoObserver(url=mongo_url, db_name='sacred'))
ex.observers.append(FileStorageObserver('logs'))

@ex.config
def cfg():
    epochs = 100

@ex.automain
def train_model(epochs):
    df = pd.read_csv('OrangeQualityData.csv')

    encoder = LabelEncoder()
    df["Color"] = encoder.fit_transform(df["Color"])
    df["Variety"] = encoder.fit_transform(df["Variety"])
    df["Blemishes"] = df["Blemishes (Y/N)"].apply(lambda x: 1 if x.startswith("Y") else 0)

    df.drop(columns=["Blemishes (Y/N)"], inplace=True)

    X = df.drop(columns=["Quality (1-5)"])
    y = df["Quality (1-5)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='sgd', loss='mse')

    history = model.fit(X_train_scaled, y_train, epochs=epochs, verbose=0, validation_data=(X_test_scaled, y_test))

    ex.log_scalar("epochs", epochs)

    ex.add_artifact(__file__)

    model.save('orange_quality_model_tf.h5')
    ex.add_artifact('orange_quality_model_tf.h5')

    for key, value in history.history.items():
        ex.log_scalar(key, value[-1])

    predictions = model.predict(X_test_scaled)
    with open('predictions_tf.json', 'w') as f:
        json.dump(predictions.tolist(), f, indent=4)
    ex.add_artifact('predictions_tf.json')

    return 'Training completed successfully'
