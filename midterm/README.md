# Handwritten Digit Recognition using MNIST

A complete Python implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits (0-9) using the MNIST dataset.

## Features

- **CNN Architecture**: Uses a 3-layer convolutional neural network for high accuracy
- **MNIST Dataset**: Trains on 60,000 training images and tests on 10,000 test images
- **Easy to Use**: Simple API for training and prediction
- **Visualization**: Includes training history plots and prediction visualizations
- **Model Saving**: Save and load trained models for later use
- **Custom Image Support**: Predict digits from your own handwritten images

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Quick Start

### 1. Train the Model

Run the main script to train the model:

```bash
python digit_recognizer.py
```

This will:
- Download the MNIST dataset automatically
- Train a CNN model for 10 epochs
- Evaluate the model on test data
- Save visualizations and the trained model
- Achieve ~99% accuracy on test data

### 2. Use the Trained Model

```python
from digit_recognizer import DigitRecognizer

# Initialize and load the trained model
recognizer = DigitRecognizer()
recognizer.load_model('digit_recognizer_model.h5')

# Predict from an image file
digit, confidence = recognizer.predict_from_file('my_digit.png')
print(f"Predicted: {digit} (Confidence: {confidence:.2f}%)")
```

## Model Architecture

```
Layer (type)                 Output Shape              Params
=================================================================
Conv2D (32 filters, 3x3)    (None, 26, 26, 32)        320
MaxPooling2D (2x2)          (None, 13, 13, 32)        0
Conv2D (64 filters, 3x3)    (None, 11, 11, 64)        18,496
MaxPooling2D (2x2)          (None, 5, 5, 64)          0
Conv2D (128 filters, 3x3)   (None, 3, 3, 128)         73,856
Flatten                     (None, 1152)              0
Dropout (0.5)               (None, 1152)              0
Dense (128 units)           (None, 128)               147,584
Dropout (0.3)               (None, 128)               0
Dense (10 units, softmax)   (None, 10)                1,290
=================================================================
Total params: 241,546
```

## Usage Examples

### Example 1: Train from Scratch

```python
from digit_recognizer import DigitRecognizer

recognizer = DigitRecognizer()
(x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
recognizer.build_model()
recognizer.train(x_train, y_train, x_test, y_test, epochs=10)
recognizer.evaluate(x_test, y_test)
recognizer.save_model()
```

### Example 2: Predict from NumPy Array

```python
import numpy as np
from digit_recognizer import DigitRecognizer

recognizer = DigitRecognizer()
recognizer.load_model('digit_recognizer_model.h5')

# Create or load your 28x28 grayscale image
image = np.random.rand(28, 28)  # Replace with actual image

digit, confidence = recognizer.predict_digit(image)
print(f"Predicted: {digit} with {confidence:.2f}% confidence")
```

### Example 3: Predict from Image File

```python
from digit_recognizer import DigitRecognizer

recognizer = DigitRecognizer()
recognizer.load_model('digit_recognizer_model.h5')

# Predict from your handwritten digit image
digit, confidence = recognizer.predict_from_file('handwritten_5.png')
```

### Example 4: Visualize Predictions

```python
from digit_recognizer import DigitRecognizer

recognizer = DigitRecognizer()
recognizer.load_model('digit_recognizer_model.h5')

(_, _), (x_test, y_test) = recognizer.load_and_preprocess_data()
recognizer.visualize_predictions(x_test, y_test, num_samples=10)
```

## Image Requirements

When using your own handwritten digit images:

1. **Format**: PNG, JPG, or any format supported by OpenCV
2. **Content**: Single digit (0-9) centered in the image
3. **Style**: Dark digit on light background (or will be inverted automatically)
4. **Resolution**: Any size (will be resized to 28x28)
5. **Background**: Clean, minimal noise for best results

## Output Files

After training, the following files are generated:

- `digit_recognizer_model.h5` - Trained model weights
- `training_history.png` - Accuracy and loss curves
- `predictions.png` - Sample predictions on test data

## Performance

- **Training Accuracy**: ~99.5%
- **Test Accuracy**: ~99.0%
- **Training Time**: ~2-3 minutes on CPU, <1 minute on GPU
- **Model Size**: ~1 MB

## Customization

### Adjust Training Parameters

```python
recognizer.train(
    x_train, y_train, 
    x_test, y_test,
    epochs=20,        # Increase for better accuracy
    batch_size=64     # Adjust based on memory
)
```

### Modify Model Architecture

Edit the `build_model()` method in `digit_recognizer.py` to:
- Add more convolutional layers
- Change filter sizes
- Adjust dropout rates
- Modify dense layer sizes

## Troubleshooting

**Issue**: Low accuracy on custom images
- **Solution**: Ensure images have dark digits on light backgrounds
- Check that digits are centered and clearly visible
- Try inverting the image colors

**Issue**: Out of memory errors
- **Solution**: Reduce batch_size in training
- Use a smaller model architecture

**Issue**: Model not saving
- **Solution**: Check write permissions in the output directory
- Ensure sufficient disk space

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Matplotlib 3.7+
- OpenCV 4.8+

## License

This project is for educational purposes.

## About MNIST Dataset

The MNIST database contains 70,000 images of handwritten digits:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels in grayscale
- Created by Yann LeCun and Corinna Cortes

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
