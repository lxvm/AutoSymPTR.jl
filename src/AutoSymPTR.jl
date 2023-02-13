"""
A package implementing the automatic symmetrized periodic trapezoidal rule (PTR)
for spectrally-accurate integration of periodic functions with the algorithm
described by [Kaye et al.](http://arxiv.org/abs/2211.12959). The main routine is
[`autosymptr`](@ref), which computes the integral of a periodic function to
within a requested error tolerance. If nontrivial symmetries are provided, the
routines return the integral on the symmetry-reduced domain, since it is
generally up to the caller to implement the mapping back to the full domain,
which is nontrivial in cases with non-scalar integrands. A simple example is

    using LinearAlgebra, AutoSymPTR
    autosymptr(x -> 1 + mapreduce(y -> cospi(2y), +, x), I(3))[1] ≈ 1
"""
module AutoSymPTR

using LinearAlgebra
using StaticArrays

export symptr, autosymptr # main routines
include("sym_ptr.jl")

end