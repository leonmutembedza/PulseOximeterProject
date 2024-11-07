
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't require a GUI
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

# Paths to the dataset directories using raw strings
train_dir = r'C:\Users\monik\Desktop\mushroom_dataset\data\train'
test_dir = r'C:\Users\monik\Desktop\mushroom_dataset\data\test'

# Image data generator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load EfficientNet model without top layers
efficientnet_base = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')

# Freeze the weights of EfficientNet layers
efficientnet_base.trainable = False

# Define input layers
input_1 = Input(shape=(150, 150, 3), name='input_1')
input_2 = Input(shape=(150, 150, 3), name='input_2')

# Process inputs with EfficientNet base
x1 = efficientnet_base(input_1)
x2 = efficientnet_base(input_2)

# Flatten the EfficientNet outputs
x1_flat = Flatten()(x1)
x2_flat = Flatten()(x2)

# CNN model for second input
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten()
])

# Process second input with the custom CNN model
y1 = cnn_model(input_1)
y2 = cnn_model(input_2)

# Concatenate the flattened outputs from EfficientNet and the CNN model
combined = Concatenate()([x1_flat, x2_flat, y1, y2])

# Fully connected layers
dense1 = Dense(512, activation='relu')(combined)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(dropout)

# Create the model
model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create a generator that combines both inputs
def combined_generator(gen1, gen2):
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        # Ensure batch sizes match
        min_batch_size = min(len(batch1[0]), len(batch2[0]))
        yield {'input_1': batch1[0][:min_batch_size], 'input_2': batch2[0][:min_batch_size]}, batch1[1][:min_batch_size]

# Callbacks for model checkpointing and early stopping
checkpoint_callback = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping_callback = EarlyStopping(patience=5, monitor='val_loss', mode='min')

# Train the model with the combined generatorfv 
history = model.fit(
    combined_generator(train_generator, train_generator),
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_data=combined_generator(test_generator, test_generator),
    validation_steps=len(test_generator),
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(combined_generator(test_generator, test_generator), steps=len(test_generator))
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Generate predictions
y_pred = model.predict(combined_generator(test_generator, test_generator), steps=len(test_generator))
y_pred_binary = (y_pred > 0.5).astype(int)

# Ensure the number of predictions matches the number of true labels
y_true = test_generator.classes[:len(y_pred_binary)]

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_plot.png')
plt.close()

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve_plot.png')
plt.close()
