# Rigid block modeling of masonry structures using mathmatical programming
This is a reproduced work of the article [Rigid block modelling of historic masonry structures using mathematical programming: a unified formulation for non‑linear time history, static pushover and limit equilibrium analysis][Francesco 2019] and [A variational rigid-block modeling approach to nonlinear elastic and kinematic analysis of failure mechanisms in historic masonry structures subjected to lateral loads][Francesco et al. 2021].

## Benchmark 1: static pushover analysis and limit analysis of a single rigid block

'mono_wall_pushover.py'

This benchmark is originally proposed by [Francesco][Francesco 2019]. We compare here the pushover curve (load multiplier versus displacement of the top right corner node) shown in [Figure 7(b)][Francesco 2019]. The reproduced result is shown below:

![pushover curve of  a single rigid block](./figures/single_block_pushover_curve.png)

The pushover curve compares fairly well with the reference, as well as the initial critical load multiplier. The slope obtained here is slightly sharper. It could be ascribed to the different solution strategy of the kinematic variables. The implementation here actually solves both [equation 29][Francesco 2019] and [equation 30][Francesco 2019]. But in the article, only [equation 29] is solved and kinematic vairables are obtained from Lagrange multipliers associated with the corresponding problem constraints.

## Benchmark 2: limit analysis of two leaves masonry walls

This benchmark is to show the impact of number of headers of two leaves masonry walls on the load multiplier of the initial configuration. The two walls studied here are shown in [Figure 11(a)][Francesco 2019] and [Figure 11(b)][Francesco 2019]. The value obtained ...

## Bemchmark 3: static pushover of masonry walls

'example_1_pushover.py'

This benchmarks is originally proposed by [Francesco][Francesco et al. 2021] in 2021. The collapse mechanism obtained is shown below:

![collapse mechanism](./figures/pushover_wall_mechanism_d200.png)

The pushover curve from the rigid contact model is:

![pushover curve of the wall panel- rigid contact model](./figures/pushover_rigid_curve.png)

[Francesco 2019]: https://link.springer.com/article/10.1007/s10518-019-00722-0
[Francesco et al. 2021]: https://onlinelibrary.wiley.com/doi/full/10.1002/eqe.3512