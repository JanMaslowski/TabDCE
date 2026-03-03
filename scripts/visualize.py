import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    train_csv_path = "data/two_moons/train.csv"
    pt_file_path = "checkpoints_const/tabdce-diffusion_metrics.pt" 
    
    plots_dir = "outputs/moons_20_instances"
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"Model not found: {pt_file_path}")

    checkpoint = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    plot_data = checkpoint["plot_data"]
    x_orig = plot_data["x_orig"]
    x_cf = plot_data["x_cf"]
    y_target = plot_data["y_target"]
    group_ids = plot_data["cf_group_ids"] 
    
    if torch.is_tensor(x_orig): x_orig = x_orig.cpu().numpy()
    if torch.is_tensor(x_cf): x_cf = x_cf.cpu().numpy()

    df_train = pd.read_csv(train_csv_path)
    X_train = df_train.iloc[:, :2].values
    y_train = df_train.iloc[:, -1].values
    unique_groups = np.unique(group_ids)
    num_plots = min(20, len(unique_groups))
    

    for i, g_id in enumerate(unique_groups[:num_plots]):
        idx = np.where(group_ids == g_id)[0]
        
        current_x_orig = x_orig[idx[0]] 
        current_cfs = x_cf[idx]         
        current_y_target = int(y_target[idx[0]])
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], alpha=0.15, c='blue', label='Class 0 (Base)')
        plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], alpha=0.15, c='red', label='Class 1 (Base)')
        
        color_target = 'red' if current_y_target == 1 else 'blue'
        color_orig = 'blue' if current_y_target == 1 else 'red'
        plt.scatter(current_x_orig[0], current_x_orig[1], c=color_orig, marker='o', 
                    edgecolor='black', linewidths=2, s=200, zorder=5, label=f'Original (Class {1 - current_y_target})')
        
        for j, cf_pt in enumerate(current_cfs):
            label = f'Cf ({len(current_cfs)} -> Class {current_y_target})' if j == 0 else ""
            
            plt.scatter(cf_pt[0], cf_pt[1], c=color_target, marker='*', 
                        edgecolor='black', linewidths=1.5, s=250, zorder=6, label=label)
            
            plt.arrow(current_x_orig[0], current_x_orig[1], 
                      cf_pt[0]-current_x_orig[0], cf_pt[1]-current_x_orig[1], 
                      color='black', alpha=0.35, head_width=0.06, length_includes_head=True, zorder=4)

        plt.title(f"To class {current_y_target}", fontsize=16, pad=15)
        plt.xlabel("f1", fontsize=12)
        plt.ylabel("f2", fontsize=12)
        plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        img_path = os.path.join(plots_dir, f"plot_{i+1:02d}_instance_{g_id}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f"[{i+1}/{num_plots}] Saved: {img_path}")

if __name__ == "__main__":
    main()