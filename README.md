# krc-stat

This repository contains code used in the experiments for the paper
> Irons NJ, Scetbon M, Pal S, Harchaoui Z. "Triangular Flows for Generative Modeling: Statistical Consistency, Smoothness Classes, and Fast Rates." AISTATS 2022. 
> [[arxiv]](https://arxiv.org/abs/2112.15595)

Our code draws heavily from the [UMNN](https://github.com/AWehenkel/UMNN) repository, and uses datasets and models from the associated paper 
> Wehenkel A, Louppe G. "Unconstrained Monotonic Neural Networks." NeurIPS 2019.
> [[arxiv]](https://arxiv.org/abs/1908.05164)
The UMNN repository in turn draws from [FFJORD](https://github.com/rtqichen/ffjord) 
and [Sylvester normalizing flows for variational inference](https://github.com/riannevdberg/sylvester-flows).

The code has been tested with Pytorch 1.1 and Python3.8.

## Files

The `figures` folder contains figures from the paper's numerical experiments that can be generated using the `plots.ipynb` notebook. This notebook draws from the folder `results`, which contains the results for each experiment. 

The notebook `run_experiments.ipynb` contains example code to load in a dataset, fit the UMNN model, and plot the results. This can be run in a few minutes on a standard cpu. For the paper, we used cluster computing to generate many replicates for each dataset over a range of sample sizes. As mentioned above, the output results are in the `results` folder. Code to implement the experiments on a slurm cluster can be provided upon request.

The folders `models` and `lib` contain libraries to generate the datasets and fit the UMNN model. These files are mostly unchanged from the UMNN repository, except for `lib/toy_data.py`, which contains code to generate additional datasets not considered in the UMNN paper.

## Datasets

Our experiments include datasets considered in the following papers:
> Wehenkel A, Louppe G. "Unconstrained Monotonic Neural Networks." NeurIPS 2019.
> [[arxiv]](https://arxiv.org/abs/1908.05164)

> Grathwohl W, Chen R, Bettencourt J, Sutskever I, Duvenaud D. "FFJORD: Free-form continuous dynamics for scalable reversible generative models." ICLR 2019.
> [[arxiv]](https://arxiv.org/abs/1810.01367)

> Chaudhuri K, Kong Z. "The expressive power of a class of normalizing flow models." AISTATS 2020.
> [[arxiv]](https://arxiv.org/abs/2006.00392)

## Cite

If you make use of this code in your own work, please cite our paper:

```
@misc{irons2021triangular,
      title={Triangular Flows for Generative Modeling: Statistical Consistency, Smoothness Classes, and Fast Rates}, 
      author={Nicholas J. Irons and Meyer Scetbon and Soumik Pal and Zaid Harchaoui},
      year={2021},
      eprint={2112.15595},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
