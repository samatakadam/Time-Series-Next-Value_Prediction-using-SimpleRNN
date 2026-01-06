import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# =====================================================
# 1. Reproducibility
# =====================================================
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 2. Data Preparation
# =====================================================
# Create a sequence (0..199)
data = np.arange(0, 200, dtype=np.float32)

# Normalize data to [0, 1]
max_val = data.max()
data_norm = data / max_val

# =====================================================
# 3. Create Sliding Window Dataset
# =====================================================
window = 5
X, y = [], []

for i in range(len(data_norm) - window):
    X.append(data_norm[i:i + window])
    y.append(data_norm[i + window])

X = np.array(X)[:, :, np.newaxis]   # (samples, timesteps, features)
y = np.array(y)

# =====================================================
# 4. Train / Test Split
# =====================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# 5. Build and Compile RNN Model
# =====================================================
model = Sequential([
    SimpleRNN(32, activation="tanh", input_shape=(window, 1)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# =====================================================
# 6. Train the Model
# =====================================================
model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    verbose=0,
    validation_data=(X_test, y_test)
)

# =====================================================
# 7. Prediction Helper Function
# =====================================================
def predict_next(sequence):
    """
    Predict the next value for a given sequence.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    assert len(sequence) == window, f"Need exactly {window} numbers"

    sequence_norm = (sequence / max_val).reshape(1, window, 1)
    pred_norm = model.predict(sequence_norm, verbose=0)[0, 0]

    return pred_norm * max_val

# =====================================================
# 8. Interactive Prediction Loop
# =====================================================
while True:
    user_input = input(
        f"\nEnter {window} numbers separated by space (or type 'quit'): "
    )

    if user_input.lower() == "quit":
        print("Exiting program.")
        break

    try:
        seq = list(map(float, user_input.split()))

        if len(seq) != window:
            print(f"❌ Please enter exactly {window} numbers.")
            continue

        prediction = predict_next(seq)
        print(f"Input: {seq} → Predicted next value: {round(prediction)}")

    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")

    except Exception as error:
        print("❌ Unexpected error:", error)
