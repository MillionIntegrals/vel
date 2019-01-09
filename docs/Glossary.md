# Glossary

* **Batch** - Most atomic step of the learning process. Usually a single forward-backward
  propagation followed by a model weight update.
  
* **Callback** - A custom piece of code that can be injected into various stages
  of training to extend the training loop.
  
* **Epoch** - A collection of batches over which we aggregate training metrics over.
  In supervised learning epoch is often equal to iterating over the whole dataset.
  
* **Metric** - A variable that we track through the learning process for introspection
  and analysis purposes.
  
* **Storage** - Object responsible for persistence of training outputs. Mostly used for 
  storing trained models and metrics collected during training.