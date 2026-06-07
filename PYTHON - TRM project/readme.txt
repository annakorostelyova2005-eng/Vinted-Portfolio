FILES:

basic_mlp.py
This file contains a simple MLP created to compare TRM performance with. MLP has similar to TRM amount of parameters.

helpers.py
This file contains various user-defined functions used for experiments' logs, calculations and visualizations.

model_training.py
This file contains full TRM and MLP training / validation procedures for both regression and classification tasks.

single_recursion.py
This file implements the core TRM recurrence for a single "reasoning step".

single_recursion_all_grad.py
The same as single_recursion.py but with tracking gradients for each hidden state update.

trm.py
This file contains the TRM model: the recursive core and some other things the paper does. It takes the single_recursion.py and uses it as a part of the TRM model.

trm_all_grad.py
The same as trm.py but with tracking gradients for each hidden state update.

update_network.py
This file builds a tiny network which learns how to update a hidden state: “given the current state and some context, how do we compute a better state?” — this is the function that TRM repeatedly reuses during its recursive reasoning.

trm_experiments.ipynb
Notebook with actual experiments and their results.

experiments_log.csv
Experiments' logging data.
