import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # Dropout pomaga w generalizacji
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_classifier(
    train_dataset, 
    clf_config: dict,
    device: torch.device,
    val_dataset=None 
):
    batch_size = clf_config.get('batch_size', 64)
    
    if val_dataset is None:
        print("[Classifier] No validation set provided. Splitting train set (80/20).")
        total_len = len(train_dataset)
        val_len = int(0.2 * total_len)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(train_dataset, [train_len, val_len])
        
        train_indices = train_subset.indices
        y_train_tensor = train_dataset.y[train_indices]
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        print("[Classifier] Using provided validation set.")
        y_train_tensor = train_dataset.y
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.X_model.shape[1]
    output_dim = max(2, train_dataset.num_classes_target) 
    
    hidden_layers = clf_config.get('hidden_layers', [64, 64])
    lr = clf_config.get('lr', 0.001)
    epochs = clf_config.get('epochs', 200)
    patience = clf_config.get('patience', 15)

    model = SimpleMLP(input_dim, hidden_layers, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    class_counts = torch.bincount(y_train_tensor)
    if len(class_counts) < 2:
        class_weights = torch.ones(output_dim).to(device)
    else:
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
    
    print(f"[Classifier] Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    print(f"[Classifier] Starting training on {device} | Patience: {patience}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, dict):
                bx = batch['x_orig'].to(device) 
                by = batch['y'].to(device)
            elif isinstance(batch, list):
                bx, by = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    bx = batch['x_orig'].to(device)
                    by = batch['y'].to(device)
                else:
                    bx, by = batch[0].to(device), batch[1].to(device)

                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(by.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        val_acc = (y_pred == y_true).mean()
        val_f1 = f1_score(y_true, y_pred, average='macro') 
        
        try:
            if output_dim == 2:
                val_auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                val_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except ValueError:
            val_auc = 0.0 

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if epochs_no_improve >= patience:
            print(f"[Classifier] Early stopping triggered at epoch {epoch+1}.")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("[Classifier] Restored best model state.")

    return model