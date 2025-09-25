# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from keras.applications.mobilenet_v2 import MobileNetV2

# Dataset parameters
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
DATASET_DIR = "violence_dataset"
CLASSES_LIST = ["non_violence", "violence"]

# Load dataset from images
features = []
labels = []

for class_index, class_name in enumerate(CLASSES_LIST):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"Folder {class_dir} does not exist!")
        continue
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping {file_path}, cannot read file.")
            continue
        img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = img / 255.0
        features.append(img)
        labels.append(class_index)

features = np.array(features)
labels = to_categorical(np.array(labels))

print(f"Total images loaded: {len(features)}")

# Split dataset into train and test
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.1, shuffle=True, random_state=42
)

# Build MobileNetV2-based CNN model
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model = Sequential([
    Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(len(CLASSES_LIST), activation='softmax')
])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=0.00005, verbose=1)

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(
    features_train, labels_train,
    validation_split=0.1,
    epochs=20,
    batch_size=16,
    shuffle=True,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate model
loss, accuracy = model.evaluate(features_test, labels_test)
print("Test Accuracy:", accuracy)

# Predictions and metrics
labels_pred = model.predict(features_test)
labels_pred_classes = np.argmax(labels_pred, axis=1)
labels_test_classes = np.argmax(labels_test, axis=1)

accuracy = accuracy_score(labels_test_classes, labels_pred_classes)
precision = precision_score(labels_test_classes, labels_pred_classes)
recall = recall_score(labels_test_classes, labels_pred_classes)
f1 = f1_score(labels_test_classes, labels_pred_classes)
roc_auc = roc_auc_score(labels_test_classes, labels_pred_classes)
conf_matrix = confusion_matrix(labels_test_classes, labels_pred_classes)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

# Save model
model.save("MobileNetV2_CNN_model.h5")
