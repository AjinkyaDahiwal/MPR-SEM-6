import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from imblearn.over_sampling import SMOTE

# 1. Load and Prepare Dataset
def load_and_prepare_data():
    # Load the credit card fraud dataset
    df = pd.read_csv('creditcard.csv')
    
    # Print basic information about the dataset
    print("\nDataset Information:")
    print(f"Total Transactions: {len(df)}")
    print(f"Fraudulent Transactions: {len(df[df['Class'] == 1])}")
    print(f"Valid Transactions: {len(df[df['Class'] == 0])}")
    print(f"Fraud Percentage: {(len(df[df['Class'] == 1]) / len(df)) * 100:.2f}%")
    
    # Visualize the class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Class')
    plt.title('Distribution of Normal vs Fraudulent Transactions')
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    plt.show()
    
    return df

# 2. Custom Dataset Class
class CreditCardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 3. Enhanced Neural Network Model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.layer4(x))
        return x

# 4. Training Function with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, device, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(100):  # Maximum 100 epochs
        # Training
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels.view(-1, 1))
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return train_losses, val_losses

# 5. Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            predicted = (outputs.cpu().numpy() > 0.5).astype(int)
            predictions.extend(predicted)
            actual.extend(labels.numpy())
    
    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(actual, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(actual, predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 6. Main execution
def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare features and labels
    X = df.drop(['Class', 'Time'], axis=1)  # Drop Time as it's not relevant
    y = df['Class']
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Apply SMOTE to handle imbalanced data (only on training set)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = CreditCardDataset(X_train_scaled, y_train_resampled)
    val_dataset = CreditCardDataset(X_val_scaled, y_val.values)
    test_dataset = CreditCardDataset(X_test_scaled, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Initialize model and training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudDetectionModel(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Load best model and evaluate
    print("\nEvaluating the model...")
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()