# train.py
from tunetools import decorator, Parameter

# This is the main function, which will be executed during training. 
# Tunetools will recognize the parameters automatically, and construct 
# grid search with the given domains. 
# "num_sample=2" means each experiment will run for 2 times.
@decorator.main(num_sample=2) 
def main(
        # Register the hyper-parameters manifest here. 
        alpha: Parameter(default=0.5, domain=[0, 0.3, 0.5]),
        beta: Parameter(default=1, domain=[0, 1, 2]),
        lr: Parameter(default=0.001, domain=[0.01, 0.001, 0.0001]),
        dataset: Parameter(default="d1", domain=["d1", "d2", "d3"]),
        model: Parameter(default="baseline", domain=["baseline", "model1", "model2"]),
        gpu: Parameter(default=0, domain=[0])
):
    # Do training here, use all the parameters...
    import random
    return {
        "result": alpha + beta + lr + (0 if model == 'baseline' else -0.5) + random.random() / 100
    }

@decorator.filtering
def filter_func(alpha, beta, lr, dataset, model, gpu):
    # Filter some parameter combinations you don't want to use.
    return dataset != 'd3'