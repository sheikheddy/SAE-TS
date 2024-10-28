from torch import nn
import numpy as np
import json
import plotly.graph_objects as go
import torch
from tqdm import tqdm
from functools import partial
from huggingface_hub import hf_hub_download

from ..steering.patch import patch_resid
from ..steering.sae import JumpReLUSAE
from ..steering.evals_utils import multi_criterion_evaluation



def steer_model(model, steer, hp, text, scale=5, batch_size=64, n_samples=128):
    toks = model.to_tokens(text, prepend_bos=True)
    toks = toks.expand(batch_size, -1)
    all_gen = []
    for _ in range(0, n_samples, batch_size):
        with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
            gen_toks = model.generate(
                toks,
                max_new_tokens=30,
                use_past_kv_cache=True,
                top_k=50,
                top_p=0.3,
                verbose=False,
            )
            all_gen.extend(model.to_string(gen_toks))
    return all_gen



PROMPTS = [
    "",
    "",
    "",
    "I think",
    "Breaking news",
    "Last night",
    "For sale",
    "The weather",
    "Dear Sir/Madam",
    "Preheat the oven",
    "It's interesting that"
    "Assistant:",
    "I went up to",
    "New study suggests",
    ]


def gen(
        model,
        steer=None,
        scale=60,
        batch_size=64,
        max_toks=32,
        n_batches=1,
        verbose=False,
        hp="blocks.12.hook_resid_post",
    ):
    if steer is not None:
        hooks = [(hp, partial(patch_resid, steering=steer, scale=scale))]
    else:
        hooks = []
    generated_tokens = []
    for prompt in tqdm(PROMPTS, disable=not verbose):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_batch = tokens.expand(batch_size, -1)
        for _ in range(n_batches):
            with model.hooks(hooks):
                gen_batch = model.generate(
                    prompt_batch,
                    max_new_tokens=max_toks - tokens.shape[-1],
                    top_k=50,
                    top_p=0.3,
                    verbose=False,
                    stop_at_eos=False,
                )
            generated_tokens.append(gen_batch)
    return torch.cat(generated_tokens, dim=0)
            

def get_feature_acts(model, sae, tokens, batch_size, hp="blocks.12.hook_resid_post"):
    assert tokens.shape[1] == 32
    all_sae_acts = torch.zeros(sae.W_dec.shape[0], device=sae.W_dec.device)
    count = 0
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp, stop_at_layer=13)
        acts = acts[hp] # shape (batch_size, len, d_model)
        acts = acts.reshape(-1, acts.shape[-1]) # shape (batch_size * len, d_model)
        acts = acts.to(torch.float32)
        sae_acts = sae.encode(acts)
        all_sae_acts += sae_acts.sum(dim=0)
        count += sae_acts.shape[0]
    return all_sae_acts / count


def get_scale(
        model,
        steer,
        loader,
        scales,
        n_batches=2,
        target_loss=4,
        hp="blocks.12.hook_resid_post",
    ):
    assert torch.allclose(torch.norm(steer), torch.tensor(1.0))
    losses = []
    for scale in scales:
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hp, partial(patch_resid, steering=steer, scale=scale))]):
                loss = model(batch['tokens'], return_type='loss')
                total_loss += loss
            if batch_idx >= n_batches - 1:
                break
        losses.append(total_loss.item() / n_batches)
        if total_loss.item()/n_batches > target_loss:
            break
    scales = scales[:len(losses)]

    # linear interpolation
    x1, x2 = scales[-2], scales[-1]
    y1, y2 = losses[-2], losses[-1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    target_scale = (target_loss - b) / m
    return target_scale


@torch.no_grad()
def get_sae(big_model=False):
    if big_model:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-res",
            filename="layer_12/width_16k/average_l0_130/params.npz",
            force_download=False)
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_12/width_16k/average_l0_82/params.npz",
            force_download=False)
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.cpu()
    return sae


@torch.no_grad()
def compute_scores(steer, model, name, criterion, hp, make_plot=True, scales=None, batch_size=64):
    if scales is None:
        scales = list(range(0, 210, 10))
    scores = []
    coherences = []
    all_texts = dict()
    products = []
    for scale in tqdm(scales):
        gen_texts = steer_model(model, steer.to(model.W_E.device), hp, "I think", scale, batch_size=batch_size)
        all_texts[scale] = gen_texts
        score, coherence = multi_criterion_evaluation(
            gen_texts,
            [
                criterion,
                "Text is coherent and the grammar is correct."
            ],
            prompt="I think",
        )
        score = [item['score'] for item in score]
        score = [(item - 1) / 9 for item in score]
        avg_score = sum(score)/len(score)
        coherence = [item['score'] for item in coherence]
        coherence = [(item - 1) / 9 for item in coherence]
        avg_coherence = sum(coherence)/len(coherence)
        scores.append(avg_score)
        coherences.append(avg_coherence)
        products.append(avg_score * avg_coherence)
    fig = go.Figure()
    fig.update_layout(
        title=f"Steering Analysis for {name}",
        xaxis_title='Scale',
        yaxis_title='Value',
        yaxis=dict(range=[0, 1])  # Set y-axis range from 0 to 1
    )
    fig.add_trace(go.Scatter(x=scales, y=coherences, mode='lines', name='Coherence'))
    fig.add_trace(go.Scatter(x=scales, y=scores, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=scales, y=products, mode='lines', name='Coherence * Score'))
    if make_plot:
        fig.show()
        fig.write_image(f"analysis_out/{name}_steer_analysis.png")
    # save all_texts as json
    if make_plot:
        with open(f"analysis_out/{name}_all_texts.json", "w") as f:
            json.dump(all_texts, f)
    return scores, coherences, products

class LinearAdapter(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, d_sae)))
        self.b = nn.Parameter(torch.zeros(d_sae))

    def forward(self, x):
        return x @ self.W + self.b
    
    @torch.no_grad()
    def compute_optimal_input(self, y):
        y_dev = y.device
        W_pinv = torch.linalg.pinv(self.W)
        x_optimal = (y.to(self.W.device) - self.b) @ W_pinv
        return x_optimal.to(y_dev)