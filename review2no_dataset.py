import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Create/Load Dataset
def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)
    
    # Generate features
    amount = np.random.lognormal(mean=5, sigma=2, size=n_samples)
    time_of_day = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)
    department_id = np.random.randint(1, 11, size=n_samples)
    vendor_id = np.random.randint(1, 1001, size=n_samples)
    transaction_type = np.random.randint(1, 5, size=n_samples)
    
    # Create some patterns for fraudulent transactions
    fraud = np.zeros(n_samples, dtype=int)
    
    # Pattern 1: High amount transactions during odd hours
    fraud[(amount > np.percentile(amount, 95)) & (time_of_day < 6)] = 1
    
    # Pattern 2: Specific department and vendor combinations
    fraud[(department_id == 5) & (vendor_id > 900)] = 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'amount': amount,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'department_id': department_id,
        'vendor_id': vendor_id,
        'transaction_type': transaction_type,
        'fraud': fraud
    })
    
    return df

# 2. Custom Dataset Class
class TransactionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 3. Neural Network Model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x

# 4. Training Function
def train_model(model, train_loader, criterion, optimizer, device):
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
    return running_loss / len(train_loader)

# 5. Main execution
def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Prepare features and labels
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = TransactionDataset(X_train_scaled, y_train.values)
    test_dataset = TransactionDataset(X_test_scaled, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model and training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudDetectionModel(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    train_losses = []
    
    for epoch in range(num_epochs):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    
    # Evaluate the model
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

if __name__ == "__main__":
    main()