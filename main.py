import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. CONFIGURATION AND PATHS
DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALID_DIR = os.path.join(DATASET_DIR, 'valid')
OUTPUT_DIR = 'metrics_and_graphs'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("TensorFlow Version:", tf.__version__)

# 2. DATA LOADING AND CLEANING
def load_and_clean_annotations(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    df_unique = df.drop_duplicates(subset=['filename']).copy()
    return df_unique

try:
    train_df = load_and_clean_annotations(os.path.join(TRAIN_DIR, '_annotations.csv'))
    valid_df = load_and_clean_annotations(os.path.join(VALID_DIR, '_annotations.csv'))
except FileNotFoundError:
    print("\n[Warning] 'dataset/' prefix not found. Checking current directory for 'train' and 'valid' folders...")
    TRAIN_DIR = 'train'
    VALID_DIR = 'valid'
    train_df = load_and_clean_annotations(os.path.join(TRAIN_DIR, '_annotations.csv'))
    valid_df = load_and_clean_annotations(os.path.join(VALID_DIR, '_annotations.csv'))

print(f" Loaded {len(train_df)} train samples and {len(valid_df)} validation samples.")

class_mapping = {'Healthy': 0, 'Malnourished': 1}
train_df['label'] = train_df['class'].map(class_mapping)
valid_df['label'] = valid_df['class'].map(class_mapping)

def parse_image(filename, label, img_dir):
    img_path = tf.strings.join([img_dir, filename], separator='/')
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0, 1]
    return img, label

def create_tf_dataset(df, img_dir, batch_size=32, shuffle=True):
    filenames = df['filename'].values
    labels = df['label'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filenames))
        
    dataset = dataset.map(lambda f, l: parse_image(f, l, img_dir), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(train_df, TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = create_tf_dataset(valid_df, VALID_DIR, batch_size=BATCH_SIZE, shuffle=False)

# 3. HIGH-ACCURACY CNN ARCHITECTURE (TRANSFER LEARNING + AUGMENTATION)
def build_advanced_model(input_shape=(224, 224, 3)):
    # Data Augmentation Layers to combat small dataset limits
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_advanced_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=6, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
]

# 4. MODEL TRAINING
print("\nStarting Advanced CNN Model Training...")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)
print(" Training complete!")

model_save_path = 'malnourished_healthy_model_advanced.keras'
model.save(model_save_path)
print(f"Optimized model weights successfully saved to {model_save_path}")

print("\nComputing evaluation metrics and generating performance graphs...")

y_true = valid_df['label'].values
y_pred_probs = model.predict(valid_dataset).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

report = classification_report(y_true, y_pred, target_names=['Healthy', 'Malnourished'])
cm = confusion_matrix(y_true, y_pred)

report_text_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_text_path, 'w') as f:
    f.write("=======================================\n")
    f.write("      CLASSIFICATION REPORT\n")
    f.write("=======================================\n")
    f.write(report)
    f.write("\n=======================================\n")
    f.write("          CONFUSION MATRIX\n")
    f.write("=======================================\n")
    f.write(np.array2string(cm))

print(f"Performance summary saved to {report_text_path}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
ax.set_title('Model Loss Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
ax.set_title('Model Accuracy Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curve.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Malnourished'], 
            yticklabels=['Healthy', 'Malnourished'],
            ax=ax, annot_kws={"size": 13, "weight": "bold"})
ax.set_title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close(fig)

# NEW: Calculate and Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2ca02c', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Classifier (AUC = 0.500)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax.set_legend(loc="lower right", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300)
plt.close(fig)

print(f"Advanced metrics and the ROC curve successfully exported to '{OUTPUT_DIR}/'!")