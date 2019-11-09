# Benchmarks

In this file I'll gather up to date benchmarking results for examples included in this repository.

Levels of hierarchy will be first task, then dataset (benchmark) and then table listing model results of
relevant metrics.

Each metric I'll try to average over six runs and provide mean and standard deviation of results.


## Generative models


### Binarized MNIST


For VAE models, I'll include upper bound for Negative Log Likelihood (NLL) for given number of importance samples (IS).


|    Model     | NLL (IS=1)  |NLL (IS=100)|NLL (IS=5000)|
|-------------:|------------:|-----------:|------------:|
|        FC VAE| 90.85 ± 0.20|87.00 ± 0.28| 86.83 ± 0.26|
|FC IWAE (k=50)|100.53 ± 0.62|82.41 ± 0.05| 80.73 ± 0.09|
|       CNN VAE| 86.47 ± 0.11|81.33 ± 0.05| 81.02 ± 0.05|
|CNN IWAE (k=5)| 88.44 ± 0.25|78.78 ± 0.05| 77.77 ± 0.06|
