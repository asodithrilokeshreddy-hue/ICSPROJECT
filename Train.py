
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from utils import load_batch
from custom_loss import focal_loss
from evaluate import calculate_metrics

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "saved_model",
    exist_ok=True
)

# ==========================
# Load Training Data
# ==========================

train_images = []
train_labels = []

for i in range(1, 6):

    images, labels = load_batch(
        f"cifar-10-batches-py/data_batch_{i}"
    )

    train_images.append(images)
    train_labels.append(labels)

x_train = np.concatenate(
    train_images,
    axis=0
)

y_train = np.concatenate(
    train_labels,
    axis=0
)

# ==========================
# Load Test Data
# ==========================

x_test, y_test = load_batch(
    "cifar-10-batches-py/test_batch"
)

# ==========================
# Normalize
# ==========================

x_train = x_train.astype(
    "float32"
) / 255.0

x_test = x_test.astype(
    "float32"
) / 255.0

# ==========================
# Resize
# ==========================

x_train = tf.image.resize(
    x_train,
    (96,96)
)

x_test = tf.image.resize(
    x_test,
    (96,96)
)

# ==========================
# One Hot Encoding
# ==========================

y_train_cat = to_categorical(
    y_train,
    10
)

y_test_cat = to_categorical(
    y_test,
    10
)

# ==========================
# MobileNet
# ==========================

base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

outputs = Dense(
    10,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=outputs
)

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss=focal_loss(),
    metrics=['accuracy']
)

model.summary()

# ==========================
# Train
# ==========================

history = model.fit(
    x_train,
    y_train_cat,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

# ==========================
# Prediction
# ==========================

pred_probs = model.predict(
    x_test
)

y_pred = np.argmax(
    pred_probs,
    axis=1
)

# ==========================
# Metrics
# ==========================

accuracy, precision, recall, f1, report = (
    calculate_metrics(
        y_test,
        y_pred
    )
)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ==========================
# Save Metrics
# ==========================

with open(
    "results/metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {accuracy}\n"
    )

    f.write(
        f"Precision: {precision}\n"
    )

    f.write(
        f"Recall   : {recall}\n"
    )

    f.write(
        f"F1 Score : {f1}\n"
    )

with open(
    "results/classification_report.txt",
    "w"
) as f:

    f.write(report)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()

# ==========================
# Accuracy Plot
# ==========================

plt.figure()

plt.plot(
    history.history['accuracy']
)

plt.plot(
    history.history['val_accuracy']
)

plt.legend(
    ['Train','Validation']
)

plt.title(
    'Accuracy'
)

plt.savefig(
    "results/accuracy.png"
)

plt.close()

# ==========================
# Loss Plot
# ==========================

plt.figure()

plt.plot(
    history.history['loss']
)

plt.plot(
    history.history['val_loss']
)

plt.legend(
    ['Train','Validation']
)

plt.title(
    'Loss'
)

plt.savefig(
    "results/loss.png"
)

plt.close()

# ==========================
# Save Model
# ==========================

model.save(
    "saved_model/mobilenet_cifar10.keras"
)

print(
    "\nProject Completed Successfully"
)
