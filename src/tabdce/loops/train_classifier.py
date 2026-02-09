import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import accuracy_score

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_classifier(dataset, clf_config: SimpleNamespace, device: torch.device):
    X_train = dataset.X_model 
    y_train = dataset.y       
    
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    
    hidden_layers = getattr(clf_config, 'hidden_layers', [30, 30])
    lr = getattr(clf_config, 'lr', 0.001)
    epochs = getattr(clf_config, 'epochs', 50)
    batch_size = getattr(clf_config, 'batch_size', 64)

    model = SimpleMLP(input_dim, hidden_layers, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Clf Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_train)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_train).float().mean().item()
    
    print(f"Klasyfikator wytrenowany. Train Accuracy: {acc:.4f}")
    return model