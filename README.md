# Decision-Making with Auto-Encoding Variational Bayes

To make decisions based on a model fit by auto-encoding variational Bayes (AEVB), practitioners often let the variational distribution serve as a surrogate for the posterior distribution. This approach yields biased estimates of the expected risk, and therefore poor decisions for two reasons. First, the model fit by AEVB may yield biased statistics relative to the underlying data distribution. Second, there may be strong discrepancies between the variational distribution and the posterior. 
We explore how fitting the variational distribution based on several objective functions other than the ELBO, while continuing to fit the generative model based on the ELBO, affects the quality of downstream decisions.
For the probabilistic principal component analysis model, we investigate how importance sampling error, as well as the biases in model parameter estimates, vary across several approximate posteriors when used as proposal distributions.
Our theoretical results suggest that a posterior approximation distinct from the variational distribution should be used for making decisions. Motivated by these theoretical results, we propose learning several approximate proposals for the best model and combining them using multiple importance sampling for decision-making. In addition to toy examples, we present a full-fledged case study of single-cell RNA sequencing. In this challenging instance of multiple hypothesis testing, our proposed approach surpasses the current state of the art.

Manuscript: https://arxiv.org/abs/2002.07217

## Install package

1. Install Python 3.7 along with PyTorch.

2. Install the package

```bash
cd decision-making-vaes
python setup.py install
```

## Run experiments

- To run the pPCA experiment, run `python simu_ppca.py`

- To run the MNIST experiment, run `python simu_mnist.py`

- To run the FDR experiment, run `python simu_fdr.py`