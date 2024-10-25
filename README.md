# SAE-TS
Improving Steering Vectors by Targeting Sparse Autoencoder Features

## Will move the code to this repo in the next few days


## Abstract
To control the behavior of language models, steering methods attempt to ensure that outputs of the model satisfy specific pre-defined properties. Adding steering vectors to the model is a promising method of model control that can be more robust than prompting and easier than fine-tuning. It can be difficult to anticipate the effects of steering vectors produced by almost all existing methods, such as CAA or the direct use of SAE latents. In our work, we address this issue by using SAEs to measure the effects of steering vectors, giving us a method that can be used to understand the causal effect of any steering vector intervention. We call our method SAE-Targeted Steering (SAE-TS) which finds steering vectors to target specific SAE features while minimizing unintended side effects. We show that overall, SAE-TS balances steering effects with coherence better than CAA and SAE feature steering, when evaluated on a range of tasks.
