TABDCE/
├── README.md
├── pyproject.toml  
├── configs/
│   └── two_moons.yaml
├── scripts/
│   └── run.py
├── notebooks/
│   └── visualize.py       
└── src/
    ├── datasets/
    │   └── dataset.py
    ├── loops/
    │   └── train.py    
    ├── models/
    │   ├── denoise_fn.py
    │   └── diffusion.py
    └── utils/
        └── metrics.py



poetry run python scripts/run.py --config configs/two_moons.yaml

export WANDB_API_KEY="cc894b19e7dc78f1805ec60df86959074051a1bc"