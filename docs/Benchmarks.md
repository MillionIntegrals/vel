# Benchmarks

In this file I'll gather up to date benchmarking results for examples included in this repository.

Levels of hierarchy will be first task, then dataset (benchmark) and then table listing model results of
relevant metrics.

Each metric I'll try to average over six runs and provide mean and standard deviation of results.


## Generative models


### Binarized MNIST


For VAE models, I'll include upper bound for Negative Log Likelihood (NLL) for given number of importance samples (IS).


| Model     | NLL (IS=1)   | NLL (IS=100) | NLL (IS=5000) |
| -----     | ----------   | ------------ | ------------- |
| FC VAE    | 90.98 ± 0.14 | 87.07 ± 0.18 | 86.93 ± 0.18  |
| CNN VAE   |
| FC IWAE   |
| CNN IWAE  |

