# RKTK

**Copyright © 2019-2024 by David K. Zhang. Released under the [MIT License][1].**

**RKTK** (Runge–Kutta toolkit) is a collection of scripts and tools, written in the [Julia][2] programming language, for discovering and optimizing new [Runge–Kutta methods][3]. This software powered my discovery of the first explicit 10<sup>th</sup>-order Runge–Kutta method with 16 stages, as described in [my undergraduate thesis][4], marking the first advance in over 40 years since [Ernst Hairer's discovery of a method with 17 stages in 1978][5].

**RKTK** is closely related to its sister repository, **[RungeKuttaToolKit.jl][6]**.

* **[RungeKuttaToolKit.jl][6]** implements the core mathematical operations of the algebraic theory of Runge–Kutta methods, such as computing Butcher weights $\Phi_t(A)$ and evaluating the gradient of the sum-of-squares residual function. **RungeKuttaToolKit.jl** is intended to be useful to all researchers studying Runge–Kutta methods and is [registered in the Julia General package registry][7], so it can be installed into any Julia environment with the command `]add RungeKuttaToolKit`.

* **RKTK** (this repository) contains parallel search and nonlinear optimization scripts that specifically target the discovery of new Runge–Kutta methods. In contrast to the pure mathematical operations in **RungeKuttaToolKit.jl**, the programs in **RKTK** establish a number of RKTK-specific conventions regarding optimization hyperparameters and file structure. Consequently, **RKTK** is not a registered Julia package.

Both **RungeKuttaToolKit.jl** and **RKTK** are designed with reproducibility as an explicit goal. The [RKTKSearch.jl][8] program has been tested to produce bit-for-bit identical results across multiple architectures (i686, x86-64, ARMv8) and multiple CPU generations (Haswell, Skylake, Rocket Lake, Zen 4).

## Getting Started

Clone this repository, then run:

```
julia -O3 --threads=<num_threads> RKTKSearch.jl <order> <num_stages>
```

Set `<num_threads>`, `<order>`, and `<num_stages>` appropriately. On processors with SMT (e.g., Intel Hyper-Threading Technology), it is worth experimenting with setting `<num_threads>` equal to the number of logical or physical processor cores. We find that results vary across CPU generations. Logical cores are favored on Haswell and Skylake, while physical cores are favored on Rocket Lake and Zen 4.

[1]: https://github.com/dzhang314/RKTK/blob/master/LICENSE
[2]: https://julialang.org/
[3]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
[4]: https://arxiv.org/abs/1911.00318
[5]: https://doi.org/10.1093/imamat/21.1.47
[6]: https://github.com/dzhang314/RungeKuttaToolKit.jl
[7]: https://juliahub.com/ui/Packages/General/RungeKuttaToolKit
[8]: https://github.com/dzhang314/RKTK/blob/master/RKTKSearch.jl
