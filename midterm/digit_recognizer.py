#!/usr/bin/env python3
"""
Handwritten Digit Recognition using MNIST Dataset
This script trains a CNN model to recognize handwritten digits (0-9)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2


class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset."""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to 0-1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"Training samples: {x_train.shape[0]}")
        print(f"Test samples: {x_test.shape[0]}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """Build a CNN model for digit recognition."""
        print("Building CNN model...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        
        return model
    
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
        """Train the model."""
        print(f"\nTraining model for {epochs} epochs...")
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """Evaluate model performance."""
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Training history saved to training_history.png")
        plt.close()
    
    def predict_digit(self, image):
        """Predict digit from a single image."""
        # Ensure image is in correct shape
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, 28, 28, 1)
        
        # Normalize if needed
        if image.max() > 1:
            image = image.astype('float32') / 255.0
        
        prediction = self.model.predict(image, verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100
        
        return digit, confidence
    
    def visualize_predictions(self, x_test, y_test, num_samples=10):
        """Visualize predictions on test samples."""
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            image = x_test[idx]
            true_label = np.argmax(y_test[idx])
            
            predicted_digit, confidence = self.predict_digit(image)
            
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].axis('off')
            
            color = 'green' if predicted_digit == true_label else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {predicted_digit}\n({confidence:.1f}%)',
                color=color, fontsize=10
            )
        
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
        print("Predictions visualization saved to predictions.png")
        plt.close()
    
    def save_model(self, filepath='digit_recognizer_model.h5'):
        """Save the trained model."""
        if self.model is None:
            print("No model to save!")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='digit_recognizer_model.h5'):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_from_file(self, image_path):
        """Predict digit from an image file."""
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Could not load image from {image_path}")
            return None, None
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert if needed (MNIST has white digits on black background)
        if img.mean() > 127:
            img = 255 - img
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Predict
        digit, confidence = self.predict_digit(img)
        
        print(f"Predicted digit: {digit} (Confidence: {confidence:.2f}%)")
        
        return digit, confidence


def main():
    """Main function to train and test the digit recognizer."""
    # Initialize recognizer
    recognizer = DigitRecognizer()
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
    
    # Build model
    recognizer.build_model()
    
    # Train model
    recognizer.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
    
    # Evaluate model
    recognizer.evaluate(x_test, y_test)
    
    # Plot training history
    recognizer.plot_training_history()
    
    # Visualize some predictions
    recognizer.visualize_predictions(x_test, y_test, num_samples=10)
    
    # Save the model
    recognizer.save_model()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nTo use the trained model:")
    print("1. Load the model: recognizer.load_model()")
    print("2. Predict from file: recognizer.predict_from_file('your_image.png')")
    print("3. Or predict from array: recognizer.predict_digit(your_image_array)")


if __name__ == "__main__":
    main()