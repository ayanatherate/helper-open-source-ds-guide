Gradient Based One sided Sampling (GOSS) explanation/ How does it contribute to better balanced predictions in Imbalanced datasets.



Gradient-based One-Side Sampling (GOSS) is a technique used in the LightGBM gradient boosting framework to improve training efficiency and generalization performance. It's specifically designed to handle imbalanced datasets, where one class is significantly more prevalent than the other. GOSS aims to address the issue of wasting computation on updating gradients for samples that contribute little to the learning process.

Here's how GOSS works:

Sample Selection:

GOSS first sorts the samples based on the absolute values of their gradients. Gradients indicate how much each sample contributes to the error, and the absolute value captures the magnitude of the error.
It divides the samples into two groups: the top samples with large gradients (indicating higher importance) and the remaining samples.
Sampling the Top Group:

The top group of samples (those with larger gradients) is retained as-is. These samples contribute significantly to the learning process and are kept unchanged.
Sampling the Bottom Group:

The bottom group of samples (those with smaller gradients) is sub-sampled. Instead of using all samples, only a subset of these samples is randomly selected. This reduces the computational cost of updating gradients for less informative samples.
Learning Process:

During each boosting iteration, LightGBM computes the gradients and Hessian for both the top group and the sub-sampled bottom group. The gradients are then used to update the model parameters.
By focusing the computational effort on the samples that have larger gradients, GOSS speeds up the training process while preserving the model's ability to learn from the informative samples. It effectively balances the trade-off between computation time and model performance. This is particularly beneficial for imbalanced datasets, where a small number of samples carry the most information.

To implement GOSS in LightGBM, you can set the boosting_type parameter to 'goss'. Additionally, you can control the fraction of sub-sampling using the subsample parameter. For example: