# AutoSymPTR.jl

[Documentation](https://lxvm.github.io/AutoSymPTR.jl/dev/)

This package provides p-adaptive quadrature for periodic integrands using the
spectrally-convergent
[Periodic Trapezoidal Rule (PTR)](https://en.wikipedia.org/wiki/Trapezoidal_rule#Periodic_and_peak_functions)
Convergence to a user-specified tolerance is automated by the algorithm.

Additionally, symmetries can be used to reduce the number of distinct quadrature
points needed for the integration, which in computational solid state physics is
known as Monkhorst-Pack integration.

## Algorithm

The algorithm of `autosymptr` is based on the ones described in
- [Kaye et al., (2023)](http://arxiv.org/abs/2211.12959)
- [Hendrik J. Monkhorst and James D. Pack Phys. Rev. B 13, 5188 – Published 15
June 1976](https://doi.org/10.1103/PhysRevB.13.5188)

## Author and Copyright

AutoSymPTR.jl was written by [Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/),
and is free/open-source software under the MIT license.

## Related packages
- [IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl)
- [HCubature.jl](https://github.com/JuliaMath/HCubature.jl)
- [Integrals.jl](https://github.com/SciML/Integrals.jl)