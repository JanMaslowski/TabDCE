import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. KONFIGURACJA I ŚCIEŻKI
    # ==========================================
    train_csv_path = "data/two_moons/test.csv"
    
    # UPEWNIJ SIĘ, że ta ścieżka prowadzi do Twojego najnowszego pliku .pt
    pt_file_path = "checkpoints_const/tabdce-diffusion_metrics.pt" 
    
    # Folder, gdzie wyląduje 20 obrazków
    plots_dir = "outputs/moons_20_instances"
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"Nie znaleziono pliku z wynikami modelu: {pt_file_path}")

    # ==========================================
    # 2. WCZYTYWANIE WYNIKÓW MODELU I TŁA
    # ==========================================
    print("Wczytywanie wygenerowanych kontrfaktów z modelu...")
    checkpoint = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    plot_data = checkpoint["plot_data"]
    
    x_orig = plot_data["x_orig"]
    x_cf = plot_data["x_cf"]
    y_target = plot_data["y_target"]
    group_ids = plot_data["cf_group_ids"] # Identyfikatory grupujące po 10 CF dla 1 punktu
    
    if torch.is_tensor(x_orig): x_orig = x_orig.cpu().numpy()
    if torch.is_tensor(x_cf): x_cf = x_cf.cpu().numpy()

    print("Wczytywanie tła (zbioru treningowego)...")
    # Niezależne od nazw kolumn (iloc)
    df_train = pd.read_csv(train_csv_path)
    X_train = df_train.iloc[:, :2].values
    y_train = df_train.iloc[:, -1].values

    # ==========================================
    # 3. GENEROWANIE 20 WYKRESÓW
    # ==========================================
    unique_groups = np.unique(group_ids)
    num_plots = min(20, len(unique_groups)) # Bierzemy 20 lub tyle, ile jest dostępnych
    
    print(f"\nRozpoczynam rysowanie {num_plots} wykresów...")

    for i, g_id in enumerate(unique_groups[:num_plots]):
        # Znajdujemy wszystkie rzędy odpowiadające tej jednej instancji (zazwyczaj 10)
        idx = np.where(group_ids == g_id)[0]
        
        # Pobieramy dane dla tej konkretnej grupy
        current_x_orig = x_orig[idx[0]] # Oryginał jest ten sam dla całej grupy
        current_cfs = x_cf[idx]         # 10 wygenerowanych punktów kontrfaktycznych
        current_y_target = int(y_target[idx[0]])
        
        plt.figure(figsize=(10, 8))
        
        # 3a. Rysowanie wyblakłego tła (żeby widzieć kształt rozkładu)
        plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], alpha=0.08, c='blue', label='Klasa 0 (Baza)')
        plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], alpha=0.08, c='red', label='Klasa 1 (Baza)')
        
        # Kolorystyka
        color_target = 'red' if current_y_target == 1 else 'blue'
        color_orig = 'blue' if current_y_target == 1 else 'red'
        
        # 3b. Rysowanie punktu oryginalnego (Wielkie Kółko)
        plt.scatter(current_x_orig[0], current_x_orig[1], c=color_orig, marker='o', 
                    edgecolor='black', linewidths=2, s=200, zorder=5, label=f'Oryginał (Klasa {1 - current_y_target})')
        
        # 3c. Rysowanie kontrfaktów i strzałek
        for j, cf_pt in enumerate(current_cfs):
            # Dodajemy etykietę do legendy tylko przy pierwszej gwiezdzie
            label = f'Kontrfakty ({len(current_cfs)} szt. -> Klasa {current_y_target})' if j == 0 else ""
            
            # Gwiazdka (Kontrfakt)
            plt.scatter(cf_pt[0], cf_pt[1], c=color_target, marker='*', 
                        edgecolor='black', linewidths=1, s=300, zorder=6, label=label)
            
            # Strzałka od oryginału do kontrfaktu
            plt.arrow(current_x_orig[0], current_x_orig[1], 
                      cf_pt[0]-current_x_orig[0], cf_pt[1]-current_x_orig[1], 
                      color='black', alpha=0.25, head_width=0.05, length_includes_head=True, zorder=4)

        # Upiększanie wykresu
        plt.title(f"Instancja #{i+1} (ID: {g_id}) | Przejście do klasy {current_y_target}", fontsize=16, pad=15)
        plt.xlabel("Cecha 1 (f1)", fontsize=12)
        plt.ylabel("Cecha 2 (f2)", fontsize=12)
        plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 3d. Zapis i czyszczenie pamięci
        img_path = os.path.join(plots_dir, f"plot_{i+1:02d}_instance_{g_id}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close() # Ekstremalnie ważne przy wielu wykresach!
        
        print(f"[{i+1}/{num_plots}] Zapisano: {img_path}")

    print(f"\n✅ Gotowe! Sprawdź folder: {plots_dir}")

if __name__ == "__main__":
    main()