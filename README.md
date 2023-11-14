# AutoSymPTR.jl

| Documentation | Build Status | Coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![][pkgeval-img]][pkgeval-url] | [![][aqua-img]][aqua-url] | [![deps-img]][deps-url] |

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

<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/AutoSymPTR.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/AutoSymPTR.jl/dev/

[action-img]: https://github.com/lxvm/AutoSymPTR.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/AutoSymPTR.jl/actions/?query=workflow:CI

[pkgeval-img]: https://juliahub.com/docs/General/AutoSymPTR/stable/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/General/AutoSymPTR

[codecov-img]: https://codecov.io/github/lxvm/AutoSymPTR.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/AutoSymPTR.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[ver-img]: https://juliahub.com/docs/AutoSymPTR/version.svg
[ver-url]: https://juliahub.com/ui/Packages/AutoSymPTR/UDEDl

[deps-img]: https://juliahub.com/docs/General/AutoSymPTR/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/AutoSymPTR?t=2