# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import itertools
import random

warnings.filterwarnings('ignore')

# 2. Load Dataset
df = pd.read_csv("../input/common-voice/cv-valid-train.csv") 
df = df[['filename', 'age', 'gender']]
data = df[df['age'].notna() & df['gender'].notna()]
data.reset_index(inplace=True, drop=True)

# 3. Clean age column
cleanup_nums = {
    "age": {
        "teens": 0.0, "twenties": 1.0, "thirties": 2.0, "fourties": 3.0,
        "fifties": 4.0, "sixties": 5.0, "seventies": 6.0, "eighties": 7.0
    }
}
data = data.replace(cleanup_nums)
data = data[:1000]  # Limit to 1000 samples

# 4. Audio Feature Extraction
ds_path = "../input/common-voice/cv-valid-train/"

def feature_extraction(filename, sampling_rate=48000):
    path = f"{ds_path}{filename}"
    features = []
    audio, _ = librosa.load(path, sr=sampling_rate)

    age = data[data['filename'] == filename].age.values[0]
    features.append(age)
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    return features

def create_df_features(orig):
    new_rows = []
    for idx, row in orig.iterrows():
        print(f"\rProcessing {idx+1}/{len(orig)}", end="")
        features = feature_extraction(row['filename'])
        new_rows.append(features)
    print()
    columns = ["label", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"] + [f"mfcc{i+1}" for i in range(20)]
    return pd.DataFrame(new_rows, columns=columns)

df_features = create_df_features(data)

# 5. Scaling Features
scaler = StandardScaler()
x = scaler.fit_transform(df_features.iloc[:, 1:].astype(float))

# 6. Encode Age Labels
encoder = LabelEncoder()
y = encoder.fit_transform(df_features.iloc[:, 0])
classes = encoder.classes_

# 7. Feature Selection (Optional)
n_features = 22
selector = SelectKBest(f_classif, k=n_features).fit(x, y)
X_new = selector.transform(x)

# 8. Train-Test Split and One-Hot Encoding
y_cat = to_categorical(y, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X_new, y_cat, test_size=0.2, random_state=0)

# 9. Build and Train Model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_new.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 10. Predict and Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 11. Confusion Matrix
def my_plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Age Group")
    plt.xlabel("Predicted Age Group")

cm = confusion_matrix(y_true, y_pred)
my_plot_confusion_matrix(cm, classes)

# 12. Show 5 Random Predictions
decoded_true = encoder.inverse_transform(y_true)
decoded_pred = encoder.inverse_transform(y_pred)
random_indices = random.sample(range(len(y_true)), 5)

print("\n5 Random Age Group Predictions:")
for idx in random_indices:
    print(f"Sample {idx}:")
    print(f"  True Age Group     : {decoded_true[idx]}")
    print(f"  Predicted Age Group: {decoded_pred[idx]}\n")

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

print("Additional Evaluation Metrics:\n")

# Ensure class names are strings
class_names = [str(c) for c in classes]

# Define labels as indices
labels = list(range(len(class_names)))

# Now use them in the report
report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
print(report)

# Macro-averaged metrics
precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)

print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1 Score : {f1:.4f}")



# Accuracy plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

## 

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import itertools
import random

warnings.filterwarnings('ignore')

# 2. Load Dataset
df = pd.read_csv("../input/common-voice/cv-valid-train.csv") 
df = df[['filename', 'age', 'gender']]
data = df[df['age'].notna() & df['gender'].notna()]
data.reset_index(inplace=True, drop=True)

# 3. Clean age column
cleanup_nums = {
    "age": {
        "teens": 0.0, "twenties": 1.0, "thirties": 2.0, "fourties": 3.0,
        "fifties": 4.0, "sixties": 5.0, "seventies": 6.0, "eighties": 7.0
    }
}
data = data.replace(cleanup_nums)
data = data[:1000]  # Limit to 1000 samples

# 4. Audio Feature Extraction
ds_path = "../input/common-voice/cv-valid-train/"

def feature_extraction(filename, sampling_rate=48000):
    path = f"{ds_path}{filename}"
    features = []
    audio, _ = librosa.load(path, sr=sampling_rate)

    gender = data[data['filename'] == filename].gender.values[0]
    features.append(gender)
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    return features

def create_df_features(orig):
    new_rows = []
    for idx, row in orig.iterrows():
        print(f"\rProcessing {idx+1}/{len(orig)}", end="")
        features = feature_extraction(row['filename'])
        features.append(row['age'])
        new_rows.append(features)
    print()
    columns = ["label", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"] + [f"mfcc{i+1}" for i in range(20)] + ["age"]
    return pd.DataFrame(new_rows, columns=columns)

df_features = create_df_features(data)

# 5. Scaling Features
scaler = StandardScaler()
x = scaler.fit_transform(df_features.iloc[:, 1:].astype(float))

# 6. Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(df_features.iloc[:, 0])
classes = encoder.classes_

# 7. Feature Selection (Optional)
n_features = 22
selector = SelectKBest(f_classif, k=n_features).fit(x, y)
X_new = selector.transform(x)

# 8. Train-Test Split and One-Hot Encoding
y_cat = to_categorical(y, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X_new, y_cat, test_size=0.2, random_state=0)

from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_new.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


# 10. Predict and Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 11. Confusion Matrix
def my_plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Gender")

cm = confusion_matrix(y_true, y_pred)
my_plot_confusion_matrix(cm, classes)

# 12. Show 5 Random Predictions
decoded_true = encoder.inverse_transform(y_true)
decoded_pred = encoder.inverse_transform(y_pred)
random_indices = random.sample(range(len(y_true)), 5)

print("\n5 Random Predictions:")
for idx in random_indices:
    print(f"Sample {idx}:")
    print(f"  True Label     : {decoded_true[idx]}")
    print(f"  Predicted Label: {decoded_pred[idx]}\n")


# 13. Additional Evaluation Metrics
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

print("Additional Evaluation Metrics:\n")
# Print classification report
report = classification_report(y_true, y_pred, target_names=classes)
print(report)

# Print macro-averaged metrics
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1 Score : {f1:.4f}")



# Accuracy plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

