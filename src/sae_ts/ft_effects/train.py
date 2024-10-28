# %%
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformer_lens import HookedTransformer

from .utils import get_sae, LinearAdapter, compute_scores
# %%

BIG_MODEL = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sae = get_sae(big_model=BIG_MODEL)

hp = "blocks.12.hook_resid_post"

# %%


def train(num_epochs, lr=1e-4):
    if BIG_MODEL:
        paths = [
            "effects/G2_9B_L12/131k_from_0",
            "effects/G2_9B_L12/131k_from_16k",
            "effects/G2_9B_L12/131k_from_32k",
        ]
    else:
        paths = [
            "effects/G2_2B_L12/65k_from_0",
            "effects/G2_2B_L12/65k_from_10k",
            "effects/G2_2B_L12/65k_from_20k",
            "effects/G2_2B_L12/65k_from_30k",
            "effects/G2_2B_L12/65k_from_40k",

            # "effects/G2_2B_L12/16k_from_0",
            # "effects/G2_2B_L12/sample_and_combine_16k",

            # "effects/G2_2B_L12/random",
            # "effects/G2_2B_L12/random_2",
            # "effects/G2_2B_L12/random_3",
            # "effects/G2_2B_L12/random_4",
            # "effects/G2_2B_L12/random_5",
        ]

    features = []
    effects = []

    for path in paths:
        features.append(torch.load(os.path.join(path, "used_features.pt")))
        effects.append(torch.load(os.path.join(path, "all_effects.pt")))

    features = torch.cat(features)
    effects = torch.cat(effects)

    # normalise features to have norm 1
    features = features / torch.norm(features, dim=-1, keepdim=True)
    n_val = 100

    val_features = features[-n_val:]
    val_effects = effects[-n_val:]
    features = features[:-n_val]
    effects = effects[:-n_val]

    dataset = TensorDataset(features, effects)
    val_dataset = TensorDataset(val_features, val_effects)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    opt = torch.optim.Adam(adapter.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        num_batches = 0

        for batch_features, batch_effects in dataloader:
            opt.zero_grad()
            batch_features = batch_features.to(device)
            batch_effects = batch_effects.to(device)
            pred = adapter(batch_features)
            loss = F.mse_loss(pred, batch_effects)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        adapter.eval()
        val_total_loss = 0
        val_num_batches = 0

        with torch.no_grad():
            for val_features, val_effects in val_dataloader:
                val_features = val_features.to(device)
                val_effects = val_effects.to(device)
                val_pred = adapter(val_features)
                val_loss = F.mse_loss(val_pred, val_effects)
                val_total_loss += val_loss.item()
                val_num_batches += 1

        scheduler.step()
        avg_val_loss = val_total_loss / val_num_batches

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


# %%
if __name__ == "__main__":
    adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])

    adapter.to(device)
    train(15, lr=2e-4)

# %%

if BIG_MODEL:
    torch.save(adapter.state_dict(), "adapter_9b_layer_12.pt")
else:
    torch.save(adapter.state_dict(), "adapter_2b_layer_12.pt")
# %%
