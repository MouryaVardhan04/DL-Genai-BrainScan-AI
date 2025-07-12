"""
Brain Tumor Detection Model Experimentation Script
This script allows you to experiment with different models, hyperparameters, and data preprocessing techniques.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import json
from datetime import datetime

class BrainTumorExperiment:
    def __init__(self, data_path="Datasets/MRI Images", model_save_path="models"):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Create directories if they don't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def load_and_preprocess_data(self, img_size=(224, 224)):
        """Load and preprocess MRI images"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        label_mapping = {}
        
        # Define tumor types
        tumor_types = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        
        for i, tumor_type in enumerate(tumor_types):
            label_mapping[tumor_type] = i
            tumor_path = os.path.join(self.data_path, tumor_type)
            
            if os.path.exists(tumor_path):
                for img_file in os.listdir(tumor_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(tumor_path, img_file)
                        try:
                            # Load and preprocess image
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, img_size)
                            img = img / 255.0  # Normalize
                            
                            images.append(img)
                            labels.append(i)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images with {len(np.unique(y))} classes")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y, label_mapping
    
    def create_cnn_model(self, input_shape=(224, 224, 3), num_classes=4):
        """Create a CNN model for brain tumor classification"""
        model = keras.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create data generators with augmentation"""
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model with given parameters"""
        print(f"Training model with {epochs} epochs, batch_size={batch_size}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Create model
        model = self.create_cnn_model(input_shape=X.shape[1:], num_classes=len(np.unique(y)))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_save_path, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history, X_val, y_val
    
    def evaluate_model(self, model, X_test, y_test, label_mapping):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Classification report
        target_names = list(label_mapping.keys())
        report = classification_report(y_test, y_pred_classes, target_names=target_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy, report, cm, y_pred
    
    def plot_training_history(self, history, save_path="experiment/training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, label_mapping, save_path="experiment/confusion_matrix.png"):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        target_names = list(label_mapping.keys())
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_experiment_results(self, accuracy, report, cm, history, label_mapping):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'label_mapping': label_mapping,
            'training_history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
        }
        
        # Save to JSON
        with open(f"experiment/results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to experiment/results_{timestamp}.json")
    
    def run_experiment(self, epochs=50, batch_size=32):
        """Run complete experiment"""
        print("Starting Brain Tumor Detection Experiment")
        print("=" * 50)
        
        # Load data
        X, y, label_mapping = self.load_and_preprocess_data()
        
        if len(X) == 0:
            print("No data found! Please check your data path.")
            return
        
        # Train model
        model, history, X_val, y_val = self.train_model(X, y, epochs, batch_size)
        
        # Evaluate model
        accuracy, report, cm, y_pred = self.evaluate_model(model, X_val, y_val, label_mapping)
        
        # Plot results
        self.plot_training_history(history)
        self.plot_confusion_matrix(cm, label_mapping)
        
        # Save results
        self.save_experiment_results(accuracy, report, cm, history, label_mapping)
        
        # Save best model
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
            model.save(os.path.join(self.model_save_path, 'brain_tumor_model.h5'))
            print(f"New best model saved with accuracy: {accuracy:.4f}")
        
        print("Experiment completed!")
        return model, accuracy

def main():
    """Main function to run the experiment"""
    experiment = BrainTumorExperiment()
    
    # Run experiment with different parameters
    print("Running experiment with default parameters...")
    model, accuracy = experiment.run_experiment(epochs=30, batch_size=32)
    
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 