import torch
from torch.utils.data import DataLoader
from tabdce.model.denoise_fn import TabularEpsModel
from tabdce.model.diffusion import MixedTabularDiffusion

def train(
    cfg,
    dataset,
):

    device = torch.device(cfg.device)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


    num_dim = dataset.num_numerical
    cat_dims = dataset.cat_cardinalities 
    x_dim = num_dim + sum(cat_dims)
    y_classes = dataset.num_classes_target

    # Model & Diffusion
    denoise_model = TabularEpsModel(
        xdim=x_dim, 
        cat_dims=cat_dims, 
        y_classes=y_classes,
        hidden=getattr(cfg, 'hidden_dim', 256)
    ).to(device)
    
    diffusion = MixedTabularDiffusion(
        denoise_fn=denoise_model,
        num_numerical=num_dim,
        num_classes=cat_dims,
        T=cfg.T,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=cfg.lr)

    denoise_model.train()
    
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x_orig = batch["x_orig"].to(device)  
            x_neigh = batch["x_neigh"].to(device)  
            y_tgt = batch["y_target"].to(device)   
            
            optimizer.zero_grad()
            loss, logs = diffusion(x_neigh, x_orig, y_tgt)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(dataloader):.4f}")
        

    diffusion.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        x_o = test_batch["x_orig"][:1].to(device)
        y_t = test_batch["y_target"][:1].to(device)
        
        cf = diffusion.sample_counterfactual(x_o, y_t)
        print("Generated CF shape:", cf.shape)
        
    return diffusion