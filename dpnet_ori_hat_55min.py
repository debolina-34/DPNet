import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
import warnings

from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint

from scipy.io import wavfile
from IPython.display import Audio
import librosa
from glob import glob
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


INPUT_LENGTH = 30225
HOP_LENGTH = 10225
NO_CLASSES = 7
SAMPLING_RATE = 20000
EPOCHS = 50

dataset_path = "D:/Debolina/AmplitudeShifted_audio_ori_hat_55min"
audio_files = glob(os.path.join(dataset_path, '**', '*'), recursive=True)
X = list()
Y = list()
hamming_window = np.hamming(INPUT_LENGTH)
for audio_file in tqdm(audio_files):
    audio, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
    for index in range(0, audio.shape[-1], INPUT_LENGTH-HOP_LENGTH):
        if index + INPUT_LENGTH > audio.shape[-1]:
            break
        x = audio[index : index + INPUT_LENGTH]
        x = x * hamming_window
        x = np.expand_dims(x, axis=0)
        y = int(os.path.basename(os.path.splitext(audio_file)[0]).split('-')[-1])
        y = np.expand_dims(y, axis=0)
        X.append(x)
        Y.append(y)
X = np.concatenate(X, axis=0)
###################################################
# Min-Max Scaler
scaler = StandardScaler()
scaler.fit(X)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
X = scaler.transform(X)
###################################################
# X = scaler.transform(X)
X = np.expand_dims(X, axis=1)
X = np.expand_dims(X, axis=3)
Y = np.concatenate(Y, axis=0)
X, Y = shuffle(X,Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

len(list(set(Y)))

len(y_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Permute, Dropout, AveragePooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Permute, DepthwiseConv2D, Dropout, AveragePooling2D, Flatten, Dense, SeparableConv2D
from keras.regularizers import l2

model = Sequential()
x=8
sr=20000
# SFEB Block
# conv 1
model.add(Conv2D(filters=8, kernel_size=(1, 9), strides=(1, 2), padding='valid', input_shape=(1, 30225, 1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

# conv 2
model.add(Conv2D(filters=64, kernel_size=(1, 5), padding='valid', strides=(1, 2), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Max Pooling 1
model.add(MaxPooling2D(pool_size=(1, int((7553/((30225/sr)*1000)*10))), strides=(1, int((7553/((30225/sr)*1000)*10)))))

# Swap axes
model.add(Permute((3, 2, 1)))

# Conv 3 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*4, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Max Pooling 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Conv 4 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*8, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Conv 5 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*8, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Max Pooling 3
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Conv 6 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*16, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Conv 7 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*16, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Max Pooling 4
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Conv 8 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*32, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Conv 9 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*32, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Max Pooling 5
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Conv 10 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*64, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))  # Dropout with rate = 0.3

# Conv 11 - Depthwise and Pointwise
model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False))
model.add(SeparableConv2D(x*64, kernel_size=(1,1), strides=(1,1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Max Pooling 6
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.2))  # Dropout with rate = 0.3

#12th conv layer
model.add(Conv2D(filters=8, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False))

#model.add(Conv2D(8, (1, 1), strides=(1,1), padding="same", use_bias=False))

model.add(BatchNormalization())
model.add(Activation('relu'))

# Final Layers
model.add(AveragePooling2D(pool_size=(2,4), strides=(2,4)))
model.add(Flatten())
model.add(Dense(units=7, activation='softmax'))

model.save('DPNet.keras')
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True),
    metrics = ['accuracy']
)

print(model.summary())


len(list(set(y_train)))

def GetLR(epoch):
    divide_epoch = np.array([EPOCHS * i for i in [0.3, 0.6, 0.9]]);
    decay = sum(epoch > divide_epoch);
    if epoch <= 30:
        decay = 1;
    return 0.1 * np.power(0.1, decay);

X[0].shape

model.predict(X)

print(dir(model))

model.inputs

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Define your cluster label mapping (ensure mapping aligns with actual KMeans clusters)
label_map = {
    0: 'Flush',
    1: 'Door',
    2: 'Basin_Tap',
    3: 'Walker_Crutch',
    4: 'Bathroom_Tap',
    5: 'No_Class',
    6: 'Shower'
}
# Step 1: Extract intermediate activations
layer_name = 'average_pooling2d'  # Use your specific layer name
intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_model.predict(X, verbose=1)

# Step 2: Flatten the feature maps (ensure shape matches)
reshaped_data = intermediate_output.reshape(intermediate_output.shape[0], -1)

# Step 3: PCA transformation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(reshaped_data)

# Step 4: KMeans clustering (labeling clusters 1 to 7)
kmeans_pca = KMeans(n_clusters=7, random_state=0)
kmeans_labels_pca = kmeans_pca.fit_predict(pca_result)

# Step 5: PCA plot
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 7))

for i in range(0, 7):
    plt.scatter(
        pca_result[kmeans_labels_pca == i, 0],
        pca_result[kmeans_labels_pca == i, 1],
        color=colors[i - 1],
        label=label_map.get(i, f'Cluster {i}'),
        edgecolor='k',
        s=50
    )

plt.title('PCA + KMeans Clustering (Before Training DPNet)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Clusters')
plt.tight_layout()
plt.savefig("pca_before_training_dpnet.png")
plt.clf()

# Step 6: t-SNE transformation
tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
tsne_result = tsne.fit_transform(reshaped_data)

# Step 7: KMeans clustering on t-SNE result
kmeans_tsne = KMeans(n_clusters=7, random_state=0)
kmeans_labels_tsne = kmeans_tsne.fit_predict(tsne_result)

# Step 8: t-SNE plot
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 7))

for i in range(0, 7):
    plt.scatter(
        tsne_result[kmeans_labels_tsne == i, 0],
        tsne_result[kmeans_labels_tsne == i, 1],
        color=colors[i - 1],
        label=label_map.get(i, f'Cluster {i}'),
        edgecolor='k',
        s=50
    )

plt.title('t-SNE + KMeans Clustering (Before Training DPNet)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.legend(title='Clusters')
plt.tight_layout()
plt.savefig("tsne_before_training_dpnet.png")
plt.clf()


history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[
    tf.keras.callbacks.LearningRateScheduler(GetLR),
    tf.keras.callbacks.ModelCheckpoint('DPNet.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
])
"""
early_stop = EarlyStopping(
    monitor='val_accuracy',  # or 'val_loss'
    patience=10,              # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # restores model weights from the epoch with the best value of the monitored quantity
    verbose=1
)

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[
        LearningRateScheduler(GetLR),
        ModelCheckpoint('DPNet.keras', monitor='val_accuracy', save_best_only=True, verbose=0),
        early_stop
    ]
)
"""
# plot accuracy graph
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig("acc_vs_epoch_dpnet.png")
plt.clf()

# plot loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig("loss_vs_epoch_dpnet.png")
plt.clf()

model.evaluate(x_train, y_train)

model.evaluate(x_test, y_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Predict class labels
y_pred = np.argmax(model.predict(x_test), axis=1)

print("Predicted shape:", y_pred.shape)
print("True labels shape:", y_test.shape)

# Class names
class_names = [
    'Flush',
    'Door',
    'Basin_Tap',
    'Walker_Crutch',
    'Bathroom_Tap',
    'No_Class',
    'Shower'
]

# Ensure all 7 classes are represented in the matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6], normalize='true')

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

plt.title("Normalized Confusion Matrix - DPNet")
plt.tight_layout()
plt.savefig("cm_dpnet.png")
plt.clf()


print(sum(y_test==4))

# Convert tensorflow model to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('DPNet.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_layer = interpreter.get_input_details()
input_shape = input_layer[0]['shape']
output_layer = interpreter.get_output_details()
accuracy = 0
for x, y in zip(x_test, y_test):
    x = np.expand_dims(x, axis=0)
    x = np.float32(x)
    interpreter.set_tensor(input_layer[0]['index'], x)
    interpreter.invoke()
    prediction = np.argmax(interpreter.get_tensor(output_layer[0]['index']))
    if y == prediction :
        accuracy += 1
print(f'DPNet TFlite Accuracy : {100*(accuracy/x_test.shape[0])}%')

def convert_bytes(bytes, to_unit):
    if to_unit == 'KB':
        return bytes / 1024
    elif to_unit == 'MB':
        return bytes / (1024 * 1024)
    elif to_unit == 'GB':
        return bytes / (1024 * 1024 * 1024)
    else:
        raise ValueError("Unsupported unit. Use 'KB', 'MB' or 'GB'.")

import os

def get_file_size(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    # Get file size in bytes
    size = os.path.getsize(file_path)
    return size

convert_bytes(get_file_size("DPNet.keras"), 'MB')

convert_bytes(get_file_size("DPNet.tflite"), 'MB')

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your model
model = load_model('DPNet.keras')

# List all layer names and their indices
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}")

# Step 1: Extract intermediate activations
layer_name = 'average_pooling2d'  # Use your specific layer name
intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_model.predict(X, verbose=1)

# Step 2: Flatten the feature maps (ensure shape matches)
reshaped_data = intermediate_output.reshape(intermediate_output.shape[0], -1)

# Step 3: PCA transformation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(reshaped_data)

# Step 4: KMeans clustering (labeling clusters 1 to 7)
kmeans_pca = KMeans(n_clusters=7, random_state=0)
kmeans_labels_pca = kmeans_pca.fit_predict(pca_result)

# Step 5: PCA plot
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 7))

for i in range(0, 7):
    plt.scatter(
        pca_result[kmeans_labels_pca == i, 0],
        pca_result[kmeans_labels_pca == i, 1],
        color=colors[i - 1],
        label=label_map.get(i, f'Cluster {i}'),
        edgecolor='k',
        s=50
    )

plt.title('PCA + KMeans Clustering (After Training DPNet)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Clusters')
plt.tight_layout()
plt.savefig("pca_after_training_dpnet.png")
plt.clf()

# Step 6: t-SNE transformation
tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
tsne_result = tsne.fit_transform(reshaped_data)

# Step 7: KMeans clustering on t-SNE result
kmeans_tsne = KMeans(n_clusters=7, random_state=0)
kmeans_labels_tsne = kmeans_tsne.fit_predict(tsne_result)

# Step 8: t-SNE plot
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 7))

for i in range(0, 7):
    plt.scatter(
        tsne_result[kmeans_labels_tsne == i, 0],
        tsne_result[kmeans_labels_tsne == i, 1],
        color=colors[i - 1],
        label=label_map.get(i, f'Cluster {i}'),
        edgecolor='k',
        s=50
    )

plt.title('t-SNE + KMeans Clustering (After Training DPNet)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.legend(title='Clusters')
plt.tight_layout()
plt.savefig("tsne_after_training_dpnet.png")
plt.clf()

