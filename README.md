# Graph Neural Networks Do Not Always Oversmooth

This repository provides the implementation for the experiments conducted in our paper:
 
Bastian Epping, Alexandre René, Moritz Helias, Michael T. Schaub [Graph Neural Networks Do Not Always Oversmooth](https://arxiv.org/abs/2406.02269) (arXiv preprint 2024)

## Installation

In an active python environment run

```bash
pip install -e PATH/TO/DIRECTORY
```

## Experiments
To perform experiments, run e.g.
```bash
cd complete_graph
python phase_diagram.py
```
The python scripts recreate the performed experiments for computationally cheaper parameter choices, such that experiments run in approximately a minute.
This was not possible for the experiments with the cora dataset, longer runtime is needed in these cases.
The parameters to recreate the experiments from the paper are given in the respective figure captions.
The scripts reproduce the experiments from the paper as follows:

* complete_graph $\rarr$ Figure 1
    * phase_diagram.py $\rarr$ panel a), b)
    * node_dist_prediction.py $\rarr$ panel c)
* cora $\rarr$ Figure 4, Figure 7
    * find_transition.py $\rarr$ fig 7
    * performance.py $\rarr$ fig 4
* Kipf_Welling_shift_operator $\rarr$ Figure 6
    * weight_variance_dependence.py $\rarr$ fig 6
* non_oversmoothing_csbm $\rarr$ Figure 2
    * condition_for_non_oversmoothing.py $\rarr$ panel a), b), c)
    * class_distance.py $\rarr$ panel d)
* performance_on_csbms $\rarr$ Figure 3, Figure 5
    * gaussian_process_GCN.py $\rarr$ fig 3 panel a), b), c)
    * finite_size_GCN.py $\rarr$ fig 3 panel d), fig 5

## Data

For our experiment in TODO, we load the Cora citation network.
We use the loading method implemented by Kipf and Welling for their paper [Graph Convolutional Network](http://arxiv.org/abs/1609.02907), ICLR 2017 (published under MIT license, Copyright (c) 2016 Thomas Kipf).
The original datasets can be downloaded from http://www.cs.umd.edu/~sen/lbc-proj/LBC.html.
The split into training and test data taken from https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016).


## Cite

Please cite our paper if you use our results or this code in your own work:

```
@article{epping2024graph,
  title={Graph Neural Networks Do Not Always Oversmooth},
  author={Epping, Bastian and Ren{\'e}, Alexandre and Helias, Moritz and Schaub, Michael T},
  journal={arXiv preprint arXiv:2406.02269},
  year={2024}
}
```

## Acknowledgements
Funded by the European Union (ERC, HIGH-HOPeS, 101039827).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency.
Neither the European Union nor the granting authority can be held responsible for them.
We also acknowledge funding by the German Research Council (DFG) within the Collaborative Research Center  “Sparsity and Singular Structures” (SfB 1481; Project A07).
