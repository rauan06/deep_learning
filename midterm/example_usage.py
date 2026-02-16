#!/usr/bin/env python3
"""
Example usage of the Digit Recognizer
"""

from digit_recognizer import DigitRecognizer
import numpy as np
import matplotlib.pyplot as plt


def example_quick_prediction():
    """Example: Load pre-trained model and make predictions."""
    print("="*60)
    print("Example: Using Pre-trained Model")
    print("="*60)
    
    # Initialize and load the trained model
    recognizer = DigitRecognizer()
    recognizer.load_model('/mnt/user-data/outputs/digit_recognizer_model.h5')
    
    # Load test data
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Preprocess
    x_test = x_test.astype('float32') / 255.0
    
    # Make predictions on a few random samples
    print("\nMaking predictions on random test samples:")
    for i in range(5):
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        true_label = y_test[idx]
        
        digit, confidence = recognizer.predict_digit(image)
        print(f"Sample {i+1}: True={true_label}, Predicted={digit}, Confidence={confidence:.2f}%")


def example_predict_custom_image():
    """Example: Predict digit from a custom image file."""
    print("\n" + "="*60)
    print("Example: Predicting from Custom Image")
    print("="*60)
    
    recognizer = DigitRecognizer()
    recognizer.load_model('/mnt/user-data/outputs/digit_recognizer_model.h5')
    
    # Example: If you have a custom image
    # digit, confidence = recognizer.predict_from_file('my_handwritten_digit.png')
    
    print("\nTo predict from your own image:")
    print("1. Draw a digit (0-9) on white paper with a dark pen")
    print("2. Take a photo or scan it")
    print("3. Use: recognizer.predict_from_file('your_image.png')")
    print("\nNote: The image should have a dark digit on light background")


def example_create_and_test_digit():
    """Example: Create a synthetic digit and test it."""
    print("\n" + "="*60)
    print("Example: Testing with Synthetic Digit")
    print("="*60)
    
    recognizer = DigitRecognizer()
    recognizer.load_model('/mnt/user-data/outputs/digit_recognizer_model.h5')
    
    # Create a simple "1" digit (vertical line)
    synthetic_digit = np.zeros((28, 28))
    synthetic_digit[5:23, 12:15] = 1  # Vertical line
    synthetic_digit[5:8, 9:12] = 1    # Top diagonal
    synthetic_digit[22:25, 8:18] = 1  # Bottom horizontal
    
    # Predict
    digit, confidence = recognizer.predict_digit(synthetic_digit)
    print(f"\nSynthetic digit predicted as: {digit} (Confidence: {confidence:.2f}%)")
    
    # Visualize
    plt.figure(figsize=(4, 4))
    plt.imshow(synthetic_digit, cmap='gray')
    plt.title(f'Predicted: {digit} ({confidence:.1f}%)')
    plt.axis('off')
    plt.savefig('/mnt/user-data/outputs/synthetic_test.png', bbox_inches='tight')
    print("Visualization saved to synthetic_test.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Digit Recognizer - Usage Examples")
    print("="*60)
    print("\nFirst, train the model by running: python digit_recognizer.py")
    print("Then run these examples!\n")
    
    # Uncomment the examples you want to run:
    
    # example_quick_prediction()
    # example_predict_custom_image()
    # example_create_and_test_digit()
    
    print("\nUncomment the examples in this file to run them!")
