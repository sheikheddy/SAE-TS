from typing import List, Callable, Union, Optional, Literal

from functools import partial
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import einops

from datasets import load_dataset

# from .evals_utils import evaluate_completions

from transformer_lens import HookedTransformer
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as tutils



def patch_resid(resid, hook, steering, scale=1):
    resid[:, :, :] = resid[:, :, :] + steering * scale
    return resid


# @torch.no_grad()
# def generate(
#     model: HookedTransformer,
#     hooks: list[tuple[str, Callable]], # includes the steering hook.
#     schedules: Optional[list[tuple[int, int]]] = None,
#     prompt = "",
#     n_samples=4,
#     batch_size=4,
#     max_new_tokens=20,
#     top_k=50,
#     top_p=0.3,
# ):
#     if schedules is None:
#         schedules = [(None, None) for _ in hooks]

#     token_step = 0

#     def counter_hook(resid, hook):
#         # keeps track of which token we're up to
#         nonlocal token_step
#         token_step += 1
#         return resid

#     def updated_hook(resid, hook, hook_fn, start, end):
#         nonlocal token_step
#         if token_step >= start and token_step <= end:
#             return hook_fn(resid, hook)
#         return resid
    
#     new_hooks = []
#     for i, (hook_layer, hook_fn) in enumerate(hooks):
#         # we modify every hook_fn to only run when the token_step is within the schedule
#         start, end = schedules[i]
#         start = start if start is not None else 0
#         end = end if end is not None else max_new_tokens + 1
#         new_hooks.append((hook_layer, partial(updated_hook, start=start, end=end, hook_fn=hook_fn)))
#     new_hooks.append(("blocks.0.hook_resid_post", counter_hook)) 

#     tokens = model.to_tokens(prompt, prepend_bos=True)
#     prompt_batch = tokens.expand(batch_size, -1)

#     results = []
#     num_batches = (n_samples + batch_size - 1) // batch_size  # Calculate number of batches

#     with model.hooks(fwd_hooks=new_hooks):
#         for _ in range(num_batches):
#             token_step = 0 # reset token_step

#             batch_results = model.generate(
#                 prompt_batch,
#                 prepend_bos=True,
#                 use_past_kv_cache=True,
#                 max_new_tokens=max_new_tokens,
#                 verbose=False,
#                 top_k=top_k,
#                 top_p=top_p,
#                 stop_at_eos=True,
#             )
#             batch_results = batch_results[:, 1:]  # cut bos
#             str_results = model.to_string(batch_results)
#             results.extend(str_results)
#     return results[:n_samples]


# # def scores_for_scales(
# #     model: HookedTransformer,

# #     hooks: list[tuple[str, Callable]],
# #     schedules: list[tuple[int, int]],

# #     prompt: str,
# #     criterions: list[str],
# #     scales: list[float],
# #     n_samples=16,
# #     insertion_pos=None,
# #     explanations=True,
# #     word_list=None,
# #     max_new_tokens=30,
# #     top_k=50,
# #     top_p=0.3,
# #     coherence_criterion="Text is coherent, the grammar is correct.",
# #     gen_log_file=None,
# # ):

# #     mean_scores = []
# #     all_scores = []

# #     for scale in scales:
# #         print("scale", scale)
# #         gen_texts = generate(
# #             model,
# #             hooks=hooks,
# #             schedules=schedules,
# #             prompt=prompt,
# #             n_samples=n_samples,
# #             max_new_tokens=max_new_tokens,
# #             top_k=top_k,
# #             top_p=top_p,
# #             )
    
    

# @torch.no_grad()
# def grid_losses(
#         model: HookedTransformer,
#         hook_point: str,
#         vector_grid: torch.Tensor,
#         ds_name="NeelNanda/c4-code-20k",
#         n_batches=50,
#         batch_size=8,
#         insertion_pos=None,
#     ):
#     """
#     Used in scores_2d
#     """
#     loss_grid = torch.zeros((vector_grid.shape[0], vector_grid.shape[1]), device=vector_grid.device)
#     print("loss computation not implemented")

#     # print(f'loading dataset: {ds_name}')
#     # data = load_dataset(ds_name, split="train")
#     # tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
#     # tokenized_data = tokenized_data.shuffle(42)
#     # loader = DataLoader(tokenized_data, batch_size=batch_size)
#     # print('dataset loaded')

#     # for i in tqdm(range(vector_grid.shape[0])):
#     #     for j in range(vector_grid.shape[1]):
#     #         steering = vector_grid[i, j]
#     #         steering = steering[None, None, :]
#     #         total_loss = 0
#     #         for batch_idx, batch in enumerate(loader):
#     #             with model.hooks(fwd_hooks=[(hook_point, partial(patch_resid,
#     #                                                             steering=steering,
#     #                                                             c=1,
#     #                                                             pos=insertion_pos,
#     #                                                             ))]):
#     #                 loss = model(batch["tokens"], return_type="loss", prepend_bos=False)
#     #                 total_loss += loss.item()
#     #             if batch_idx >= n_batches-1:
#     #                 break
#     #         loss_grid[i, j] = total_loss / n_batches
#     return loss_grid


# @torch.no_grad()
# def scores_2d(
#     model: HookedTransformer,
#     hook_point: str,
#     steering_vectors: list,
#     prompt: str,
#     criterions: list[str],
#     scales: list[float],
#     n_samples=10,
#     batch_size=4,
#     model_name="gpt-3.5-turbo", # "gpt-4o"
#     max_new_tokens=30,
#     top_k=50,
#     top_p=0.3,
#     coherence_criterion="Text is coherent, the grammar is correct.",
#     gen_log_file=None,
#     ):
#     assert len(steering_vectors) == len(criterions) == 2

#     v1, v2 = steering_vectors
#     device = v1.device

#     vector_grid = torch.zeros((len(scales), len(scales), v1.shape[-1]), device=device)
#     for i, s1 in enumerate(scales):
#         for j, s2 in enumerate(scales):
#             vector_grid[i, j] = v1 * s1 + v2 * s2

#     loss_grid = grid_losses(model, hook_point, vector_grid)
#     print("loss grid")
#     print(loss_grid)

#     score_grid_1 = torch.zeros_like(loss_grid, device=device)
#     score_grid_2 = torch.zeros_like(loss_grid, device=device)
#     coherence_grid = torch.zeros_like(loss_grid, device=device)

#     all_gen_texts = []

#     for i in range(len(scales)):
#         for j in range(len(scales)):
#             print(f'evaluating ({i}, {j})')
#             steering_vector = vector_grid[i, j]
#             hooks = [(hook_point, partial(patch_resid, steering=steering_vector, scale=1))]

#             gen_texts = generate(
#                 model,
#                 hooks=hooks,
#                 prompt=prompt,
#                 n_samples=n_samples,
#                 max_new_tokens=max_new_tokens,
#                 batch_size=batch_size,
#                 top_k=top_k,
#                 top_p=top_p,
#             )

#             print(gen_texts)
#             eval_1 = evaluate_completions(gen_texts, criterion=criterions[0], prompt=prompt, verbose=False, model=model_name)
#             eval_2 = evaluate_completions(gen_texts, criterion=criterions[1], prompt=prompt, verbose=False, model=model_name)
#             coherence = evaluate_completions(gen_texts, criterion=coherence_criterion, prompt=prompt, verbose=False, model=model_name)
#             scores_1 = [e['score'] for e in eval_1]
#             scores_2 = [e['score'] for e in eval_2]
#             coherence_scores = [e['score'] for e in coherence]

#             all_gen_texts.append(
#                 {
#                     "texts": gen_texts,
#                     "scales": [scales[i], scales[j]],
#                     "scores_1": scores_1,
#                     "scores_2": scores_2,
#                     "coherence_scores": coherence_scores,
#                 }
#             )

#             mean_1 = sum(scores_1) / len(scores_1)
#             mean_2 = sum(scores_2) / len(scores_2)
#             mean_coherence = sum(coherence_scores) / len(coherence_scores)
#             score_grid_1[i, j] = mean_1
#             score_grid_2[i, j] = mean_2
#             coherence_grid[i, j] = mean_coherence
#             print(criterions[0], mean_1)
#             print(criterions[1], mean_2)
#             print(coherence_criterion, mean_coherence)
    
#     if gen_log_file is not None:
#         with open(gen_log_file, "w") as f:
#             f.write(json.dumps(all_gen_texts))
        
#     return score_grid_1.cpu(), score_grid_2.cpu(), loss_grid.cpu(), coherence_grid.cpu()