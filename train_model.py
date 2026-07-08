"""
train_model.py
Trains the CNN from Notebook 1A on the digits dataset and saves it to disk
so the Streamlit app (app.py) can load it without retraining.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

digits = load_digits()
X, y = digits.images, digits.target

x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1765, random_state=SEED, stratify=y_train_full
)

x_train = np.expand_dims(x_train.astype("float32") / 16.0, -1)
x_val = np.expand_dims(x_val.astype("float32") / 16.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 16.0, -1)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, monitor="val_loss"),
]

model.fit(x_train, y_train_cat, validation_data=(x_val, y_val_cat),
          epochs=40, batch_size=32, callbacks=callbacks, verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

model.save("digit_cnn_model.keras")
print("Saved model to digit_cnn_model.keras")
