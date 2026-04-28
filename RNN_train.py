import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import joblib

# 1. Prepare Data
def prepare_data(df, feature_col='Close', sequence_length=20):
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(df[[feature_col]])

    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    return X, y, scaler


# 2. Build Model
def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# 3. Train Model
def train_lstm(df):
    X, y, scaler = prepare_data(df)

    model = build_lstm_model((X.shape[1], X.shape[2]))

    model.fit(X, y, epochs=10, batch_size=32)

    # Save model + scaler
    model.save("lstm_model.h5")
    joblib.dump(scaler, "scaler.save")

    print("Model and scaler saved!")

    return model, scaler



# 4. Load Model (for inference)
def load_lstm():
    model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.save")

    return model, scaler


# 5. Predict Next Value
def predict_next(model, scaler, df, sequence_length=20):
    last_data = df[['Close']].tail(sequence_length)

    scaled = scaler.transform(last_data)

    X = np.array([scaled])

    prediction = model.predict(X)

    # Inverse scale
    prediction = scaler.inverse_transform(prediction)

    return prediction[0][0]