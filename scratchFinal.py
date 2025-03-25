import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.costs = []
        
    def normalize_features(self, X):
        """
        Custom normalization: (x - mean) / std
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8), mean, std
    
    def sigmoid(self, z):
        """
        Sigmoid activation function with clipping to prevent overflow
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    def fit(self, X, y):
        """
        Train the logistic regression model
        """
        # Normalize features
        X_normalized, self.mean, self.std = self.normalize_features(X)
        
        # Get dimensions
        n_samples, n_features = X_normalized.shape
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward propagation
            z = np.dot(X_normalized, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Calculate cost
            cost = (-1/n_samples) * np.sum(
                y * np.log(predictions + 1e-10) + 
                (1-y) * np.log(1 - predictions + 1e-10)
            )
            
            # Calculate gradients
            dz = predictions - y
            dw = (1/n_samples) * np.dot(X_normalized.T, dz)
            db = (1/n_samples) * np.sum(dz)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store cost
            if i % 100 == 0:
                self.costs.append(cost)
                print(f'Cost after iteration {i}: {cost}')
    
    def predict_proba(self, X):
        """
        Predict probability of class 1
        """
        # Normalize input using training mean and std
        X_normalized = (X - self.mean) / (self.std + 1e-8)
        z = np.dot(X_normalized, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        """
        return (self.predict_proba(X) >= threshold).astype(int)

def load_and_prepare_data():
    """
    Load and prepare the credit card fraud dataset
    """
    print("Loading data...")
    df = pd.read_csv('creditcard.csv')
    
    # Remove Time column and separate features/target
    X = df.drop(['Class', 'Time'], axis=1).values
    y = df['Class'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    """
    Calculate and print evaluation metrics
    """
    # Calculate confusion matrix elements
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)-0.02
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.yticks([0, 1], ['Normal', 'Fraud'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_training_cost(costs):
    """
    Plot the training cost over iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(costs) * 100, 100), costs)
    plt.title('Training Cost Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Create and train model
    print("\nTraining model...")
    model = CustomLogisticRegression(learning_rate=0.01, num_iterations=500)
    
    model.fit(X_train, y_train)
    
    # Plot training cost
    plot_training_cost(model.costs)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate model
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()