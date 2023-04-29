import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from werkzeug.utils import secure_filename

#Variables - training directory and Image size.
data_dir = 'training/felidae'
IMG_SIZE = 64

def load_images(species, data_dir):
    data = []
    labels = []
    for sp in species:
        sp_dir = os.path.join(data_dir, sp)
        img_count = 0
        for img_file in os.listdir(sp_dir):
            img_path = os.path.join(sp_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img_array = np.asarray(img_resized)
                img_array = img_array.reshape((IMG_SIZE, IMG_SIZE, 3))
                data.append(img_array)
                labels.append(species.index(sp))
                img_count += 1
            except Exception as e:
                print(f"Error processing image: {img_path}")
                print(f"Error details: {e}")
        print(f"{sp}: {img_count} images loaded")
    return np.array(data), np.array(labels)

species = ['Cheetah', 'Leopard', 'Lion', 'Puma', 'Tiger']
train_data, train_labels = load_images(species, data_dir)
print(f"Total training images: {len(train_data)}")

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model.fit(train_generator, epochs=90, validation_data=val_generator, callbacks=[early_stop])

test_data = []
test_labels = []

for sp in species:
    sp_dir = os.path.join(data_dir, sp)
    for img_file in os.listdir(sp_dir):
        img_path = os.path.join(sp_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_array = np.asarray(img_resized)
            img_array = img_array.reshape((IMG_SIZE, IMG_SIZE, 3))
            test_data.append(img_array)
            test_labels.append(species.index(sp))
        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Error details: {e}")

test_data = np.array(test_data) / 255.0
test_labels = np.array(test_labels)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')

model.save('my_model.h5')
