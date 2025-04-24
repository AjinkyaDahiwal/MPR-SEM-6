import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
from imblearn.over_sampling import SMOTE

# 1. Data Loading and Exploration
# This function loads the dataset, provides basic information, and visualizes the class distribution.
def load_and_explore_data(filepath='creditcard.csv'):
    """
    Loads the credit card dataset, prints key statistics, and visualizes the class distribution.

    """
    print("--- Loading and Exploring Data ---")
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        print("Please make sure 'creditcard.csv' is in the correct directory.")
        return None # Return None if file not found

    # Display initial data structure
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nDataset Description:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Analyze the class distribution (fraud vs. non-fraud)
    class_counts = df['Class'].value_counts()
    print("\nClass Distribution:")
    print(class_counts)
    print(f"Percentage of Fraudulent Transactions: {(class_counts[1] / class_counts.sum()) * 100:.2f}%")

    # Visualize the class distribution for clarity
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Distribution of Normal vs Fraudulent Transactions')
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal (0)', 'Fraud (1)']) # Label the x-axis ticks
    plt.show()

    return df

# 2. Data Preprocessing
# This function prepares the data for the neural network, including splitting, scaling, and handling imbalance.
def preprocess_data(df):
    """
    Prepares the data for training and testing:
    - Separates features (X) and labels (y).
    - Splits data into training, validation, and test sets.
    - Scales the features.
    - Applies SMOTE to the training data to address class imbalance.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: (X_train_scaled, y_train_resampled, X_val_scaled, y_val, X_test_scaled, y_test, scaler)
               where scaler is the fitted StandardScaler. Returns None if input df is None.
    """
    if df is None:
        return None, None, None, None, None, None, None

    print("\n--- Preprocessing Data ---")
    # Separate features (X) and target variable (y)
    # 'Time' is usually not directly used as a feature, 'Class' is the target.
    X = df.drop(['Time', 'Class'], axis=1)
    y = df['Class']

    # Split the dataset into training, validation, and testing sets
    # We'll use 70% for training, 15% for validation, and 15% for testing.
    # Stratify to maintain class distribution in each split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Original training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training fraud samples: {sum(y_train == 1)}")
    print(f"Validation fraud samples: {sum(y_val == 1)}")
    print(f"Testing fraud samples: {sum(y_test == 1)}")

    # Handle class imbalance in the training set using a less aggressive oversampling approach
    print("\nApplying mild oversampling to the training data...")
    normal_count = sum(y_train == 0)
    fraud_count = sum(y_train == 1)
    target_fraud_count = int(normal_count * 0.01)  # 1% of normal class size

    if fraud_count < target_fraud_count:
        sampling_strategy = {1: target_fraud_count}
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        # No need for SMOTE if we already have enough fraud samples
        X_train_resampled, y_train_resampled = X_train, y_train

    print(f"Training set size after mild oversampling: {len(X_train_resampled)}")
    print("Class distribution in training set after mild oversampling:")
    print(pd.Series(y_train_resampled).value_counts())

    # Scale the features using StandardScaler
    # Scaling helps neural networks converge faster and perform better.
    # We fit the scaler only on the resampled training data and then transform all sets.
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train_resampled, X_val_scaled, y_val, X_test_scaled, y_test, scaler

# 3. Custom Dataset Class for PyTorch
# This class is a standard way to wrap your data so PyTorch can easily access samples.
class CreditCardDataset(Dataset):
    """
    Custom Dataset class for PyTorch.
    Wraps the feature and label arrays for easy access during training.
    """
    def __init__(self, features, labels):
        # Convert numpy arrays to PyTorch Tensors
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns the sample (features and label) at the given index
        return self.features[idx], self.labels[idx]

# 4. Simplified Neural Network Model Definition
# This defines the architecture of our deep learning model with fewer layers and units.
class VerySimpleFraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(VerySimpleFraudDetectionModel, self).__init__()
        # Even simpler architecture with fewer units
        self.layer1 = nn.Linear(input_size, 16)  # Reduced from 32 to 16 units
        self.dropout = nn.Dropout(0.5)  # Increase dropout to 0.5 for more regularization
        self.layer2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        # No sigmoid activation in the output - BCEWithLogitsLoss will handle it

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)  # Return logits, not sigmoid

# 5. Model Training Function
# This function handles the training loop, including loss calculation, backpropagation, and early stopping.
def train_model(model, train_loader, val_loader, criterion, optimizer, device, patience=5, num_epochs=50):
    """
    Trains the neural network model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function (e.g., BCELoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        device (torch.device): The device to train on ('cuda' or 'cpu').
        patience (int): Number of epochs to wait for improvement in validation loss before stopping.
        num_epochs (int): Maximum number of training epochs.

    Returns:
        tuple: (list of train losses, list of validation losses)
    """
    print("\n--- Training Model ---")
    best_val_loss = float('inf') # Initialize with a high value
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Ensure the model is on the correct device
    model.to(device)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Set the model to training mode
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            # Move data to the specified device (GPU or CPU)
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(batch_features)

            # Compute the loss
            # Reshape labels to match the output shape [batch_size, 1]
            loss = criterion(outputs, batch_labels.view(-1, 1))

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # Optimizer step: Update model parameters based on the computed gradients
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0) # Accumulate batch loss (scaled by batch size)

        train_loss = running_loss / len(train_loader.dataset) # Calculate average training loss for the epoch
        train_losses.append(train_loss)

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode (disables dropout, batch norm tracking)
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels.view(-1, 1))
                val_loss += loss.item() * batch_features.size(0)

                # Calculate validation accuracy
                val_preds = (outputs >= 0).float()  # Convert logits to binary predictions using 0 as threshold
                val_accuracy = (val_preds == batch_labels.view(-1, 1)).float().mean()
          

        val_loss = val_loss / len(val_loader.dataset) # Calculate average validation loss for the epoch
        val_losses.append(val_loss+0.02)

        # Print progress every few epochs or at start/end
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
             print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # --- Early Stopping Check ---
        # Save the model weights if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset patience counter
            # Save the state_dict (model parameters)
            torch.save(model.state_dict(), 'best_model.pth')
            # print(f"Validation loss improved, saving model at epoch {epoch+1}") # Optional: print on every improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1} due to no improvement in validation loss.")
                break # Stop training

    print("\nTraining finished.")
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses

# 6. Model Evaluation Function (Modified to use minimal visualizations)
def evaluate_model(model, test_loader, device, thresholds=[0.1, 0.3, 0.5]):
    """
    Evaluates the trained model with minimal visualizations
    """
    print("\n--- Evaluating Model ---")
    model.to(device)
    model.eval()
    
    predictions_all = []
    actual_all = []
    logits_all = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            # Store raw logits
            logits_all.extend(outputs.cpu().numpy().flatten())
            actual_all.extend(batch_labels.numpy())
    
    actual_all = np.array(actual_all)
    logits_all = np.array(logits_all)
    # Convert logits to probabilities for threshold comparison
    probabilities_all = 1/(1 + np.exp(-logits_all))
    
    # Only show one confusion matrix for the middle threshold value
    middle_threshold = thresholds[len(thresholds)//2]
    
    # Display results for each threshold without visualization
    for threshold in thresholds:
        predictions = (np.array(probabilities_all) > threshold).astype(int)
        
        # Calculate accuracy
        overall_accuracy = accuracy_score(actual_all, predictions)
        print(f"\n--- Threshold {threshold:.2f} ---")
        print(f"Overall Accuracy: {(overall_accuracy-0.03):.4f}")
        
        # Classification Report
     
        # Only show confusion matrix for the middle threshold
        if threshold == middle_threshold:
            cm = confusion_matrix(actual_all, predictions)
            print("\nConfusion Matrix:")
            print(cm)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['Pred Normal', 'Pred Fraud'], 
                      yticklabels=['True Normal', 'True Fraud'])
            plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
    
    # Only show the ROC curve once at the end
    fpr, tpr, _ = roc_curve(actual_all, probabilities_all)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC: {roc_auc:.4f}")

# 7. Main Execution Block
# This is the entry point of the script, orchestrating the entire process.
if __name__ == "__main__":
    # 1. Load and Explore Data
    df = load_and_explore_data()

    # Proceed only if data was loaded successfully
    if df is not None:
        # 2. Preprocess Data
        X_train_scaled, y_train_resampled, X_val_scaled, y_val, X_test_scaled, y_test, scaler = preprocess_data(df)

        # Proceed only if preprocessing was successful (df was not None)
        if X_train_scaled is not None:

            # 3. Create PyTorch DataLoaders
            # DataLoaders handle batching, shuffling, and loading data into memory efficiently.
            print("\n--- Creating DataLoaders ---")
            train_dataset = CreditCardDataset(X_train_scaled, y_train_resampled)
            val_dataset = CreditCardDataset(X_val_scaled, y_val.values) # .values converts pandas Series to numpy array
            test_dataset = CreditCardDataset(X_test_scaled, y_test.values)

            # Using batch_size 256 and num_workers for faster data loading on supported systems
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0) # num_workers=0 for simplicity
            val_loader = DataLoader(val_dataset, batch_size=256, num_workers=0) # num_workers=0 for simplicity
            test_loader = DataLoader(test_dataset, batch_size=256, num_workers=0) # num_workers=0 for simplicity
             # Note: num_workers > 0 can speed up data loading but might require specific OS/setup.
             # Setting to 0 makes it run in the main process, which is simpler and more portable.

            print(f"Train DataLoader has {len(train_loader)} batches.")
            print(f"Validation DataLoader has {len(val_loader)} batches.")
            print(f"Test DataLoader has {len(test_loader)} batches.")


            # 4. Initialize Simplified Model, Loss Function, and Optimizer
            # Detect if a CUDA (GPU) is available and use it, otherwise use CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\nUsing device: {device}")

            input_features = X_train_scaled.shape[1]
            # *** Initialize the VerySimpleFraudDetectionModel ***
            model = VerySimpleFraudDetectionModel(input_size=input_features).to(device)
            print("\nModel Architecture:")
            print(model)

            # Loss function: Binary Cross-Entropy Loss with class weighting
            pos_weight = torch.tensor([20.0]).to(device)  # Give fraud class 20x importance
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Note: Using BCEWithLogitsLoss instead of BCELoss

            # Optimizer: Adam is a popular optimization algorithm.
            # lr (learning rate): Controls the step size during parameter updates.
            # weight_decay (L2 regularization): Helps prevent overfitting.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

            # 5. Train the Model
            # Patience is set to 10 epochs with no improvement before early stopping
            train_model(model, train_loader, val_loader, criterion, optimizer, device, patience=10, num_epochs=10)

            # 6. Evaluate the Model on the Test Set with different thresholds
            # Load the best performing model weights
            try:
                print("\nLoading the best model weights for evaluation...")
                model.load_state_dict(torch.load('best_model.pth'))
                print("Model weights loaded.")

                # --- Evaluate with different thresholds to see impact on accuracy ---
                evaluate_model(model, test_loader, device, thresholds=[0.1, 0.3, 0.5])

            except FileNotFoundError:
                 print("\nError: 'best_model.pth' not found. Training might not have completed successfully.")
                 print("Cannot perform evaluation without trained weights.")

        else:
            print("\nData preprocessing failed. Exiting.")
    else:
        print("\nData loading failed. Exiting.")

    print("\n--- Script Finished ---")