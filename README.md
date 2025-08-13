# Logistic Regression Implementation from Scratch

## Overview
This project implements a Logistic Regression classifier from scratch using the Perceptron algorithm. The implementation demonstrates the fundamental concepts of binary classification and linear decision boundaries in machine learning.

## Features
- **Binary Classification**: Implements a two-class classification problem
- **Synthetic Dataset Generation**: Uses scikit-learn's `make_classification` to create a controlled dataset
- **Perceptron Algorithm**: Implements the classic Perceptron learning rule
- **Visualization**: Includes comprehensive plotting of data points and decision boundary
- **From Scratch Implementation**: No external ML libraries used for the core algorithm

## Requirements
```python
numpy
matplotlib
scikit-learn (for dataset generation only)
```

## Installation
```bash
pip install numpy matplotlib scikit-learn
```

## Usage

### 1. Dataset Generation
The code generates a synthetic binary classification dataset with:
- 100 samples
- 2 features
- 2 classes
- Controlled separation between classes for clear visualization

### 2. Perceptron Algorithm
The core algorithm implements the Perceptron learning rule:
- Adds bias term (intercept) to input features
- Initializes weights randomly
- Updates weights using the Perceptron update rule
- Uses step function for binary classification

### 3. Decision Boundary Visualization
- Plots the original data points colored by class
- Draws the learned decision boundary as a red line
- Shows how well the algorithm separates the two classes

## Code Structure

### Main Functions

#### `perceptron(X, y)`
- **Input**: Feature matrix X and target vector y
- **Output**: Intercept (bias) and coefficients (weights)
- **Algorithm**: Implements Perceptron learning with 1000 iterations
- **Learning Rate**: 0.1

#### `step(z)`
- **Input**: Linear combination z
- **Output**: Binary classification (0 or 1)
- **Function**: Step function that returns 1 if z > 0, else 0

### Key Variables
- `intercept_`: Bias term (w₀)
- `coef_`: Feature weights (w₁, w₂)
- `m`: Slope of decision boundary
- `b`: Y-intercept of decision boundary

## Mathematical Foundation

The decision boundary is defined by the equation:
```
w₀ + w₁x₁ + w₂x₂ = 0
```

Where:
- w₀ is the bias term
- w₁, w₂ are the feature weights
- x₁, x₂ are the input features

The step function provides binary classification:
```
f(z) = 1 if z > 0, else 0
```

## Visualization

The code generates two main plots:
1. **Data Scatter Plot**: Shows the original data points colored by class
2. **Decision Boundary Plot**: Overlays the learned decision boundary on the data

## Learning Process

1. **Initialization**: Weights are initialized to ones
2. **Training Loop**: 1000 iterations of weight updates
3. **Random Sampling**: Each iteration uses a random training example
4. **Weight Update**: Weights are updated using the Perceptron rule:
   ```
   w = w + learning_rate * (true_label - predicted_label) * features
   ```

## Limitations

- **Linear Separability**: Perceptron only works for linearly separable data
- **Binary Classification**: Limited to two-class problems
- **No Convergence Guarantee**: May not converge if data is not linearly separable

## Applications

This implementation is useful for:
- Educational purposes in understanding ML fundamentals
- Simple binary classification tasks
- Demonstrating linear decision boundaries
- Learning the Perceptron algorithm

## Future Improvements

- Add convergence checking
- Implement different activation functions
- Support for multi-class classification
- Regularization techniques
- Cross-validation
- Performance metrics (accuracy, precision, recall)

## Author
Created as part of machine learning projects to understand fundamental concepts.

## License
This project is for educational purposes.
