import torch
import torch.nn as nn
import torch.nn.functional as F
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
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ClassifierOHEWrapper(nn.Module):
    def __init__(self, base_classifier: nn.Module, cat_cardinalities: list):
        super().__init__()
        self.base_classifier = base_classifier
        self.cat_cardinalities = cat_cardinalities

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_parts = []
        if x_cat is not None and x_cat.shape[1] > 0:
            for i, card in enumerate(self.cat_cardinalities):
                idx_col = x_cat[:, i].long()
                oh = F.one_hot(idx_col, num_classes=card).float()
                cat_parts.append(oh)
        
        if len(cat_parts) > 0:
            x_flat = torch.cat([x_num] + cat_parts, dim=-1)
        else:
            x_flat = x_num
            
        return self.base_classifier(x_flat)

def train_classifier(
    train_dataset, 
    clf_config: dict,
    device: torch.device,
    val_dataset=None 
):
    batch_size = clf_config.get('batch_size', 64)
    
    if val_dataset is None:
        total_len = len(train_dataset)
        val_len = int(0.2 * total_len)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(train_dataset, [train_len, val_len])
        
        train_indices = train_subset.indices
        y_train_tensor = train_dataset.y[train_indices]
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        y_train_tensor = train_dataset.y
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.num_numerical + sum(train_dataset.cat_cardinalities)
    output_dim = max(2, train_dataset.num_classes_target) 
    
    hidden_layers = clf_config.get('hidden_layers', [64, 64])
    lr = clf_config.get('lr', 0.001)
    epochs = clf_config.get('epochs', 200)
    patience = clf_config.get('patience', 15)

    base_model = SimpleMLP(input_dim, hidden_layers, output_dim)
    model = ClassifierOHEWrapper(base_model, train_dataset.cat_cardinalities).to(device)
    
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
                bx_num = batch['x_num_orig'].to(device) 
                bx_cat = batch['x_cat_orig'].to(device)
                by = batch['y'].to(device)
            else:
                raise ValueError("No dictionary")
            
            optimizer.zero_grad()
            logits = model(bx_num, bx_cat)
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
                bx_num = batch['x_num_orig'].to(device)
                bx_cat = batch['x_cat_orig'].to(device)
                by = batch['y'].to(device)

                logits = model(bx_num, bx_cat)
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