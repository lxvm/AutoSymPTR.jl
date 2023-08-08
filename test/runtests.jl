using LinearAlgebra
using Test

using StaticArrays
using FFTW

using AutoSymPTR
using AutoSymPTR: Basis, PTR, MonkhorstPack, pquadrature

"""
    cube_automorphisms(d::Integer)

return a generator of the symmetries of the cube in `d` dimensions including the
identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C
n_permutations(n::Integer) = factorial(n)


@testset "AutoSymPTR" begin

    @testset "PTR" begin
        @testset "volume" begin
            f(x) = 1.0
            buffer = Float64[]
            for npt in (30,), dims in 1:3
                B = rand(dims,dims)
                dom = Basis(B)
                rule = PTR(Float64, Val(dims), npt)
                @test abs(det(B)) ≈ rule(f, dom) ≈ rule(f, dom, buffer)
            end
        end

        @testset "oscillations" begin
            f(x) = 1.0 + sum(cos, x)
            buffer = Float64[]
            for npt in (30,), dims in 1:3
                B = 2pi*I(dims)
                dom = Basis(B)
                rule = PTR(Float64, Val(dims), npt)
                @test abs(det(B)) ≈ rule(f, dom) ≈ rule(f, dom, buffer)
            end
        end

        @testset "convergence" begin
            f(x, ω=0.3, η=1e-1) = inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
            for dims in 1:2
                dom = Basis(I(dims))
                ref = PTR(Float64, Val(dims), 100)(f, dom)
                npts = (10, 20, 40, 80) # geometric progression for asymptotic behavior
                sols = Vector{ComplexF64}(undef, length(npts))
                for (i,npt) in enumerate(npts)
                    sols[i] =  PTR(Float64, Val(dims), npt)(f, dom)
                end
                @test issorted(reverse(norm.(sols .- ref))) # check errors decrease
                # error for PTR goes like err_n ~< exp(-aηn)
                lerr = log10.(norm.(sols .- ref))
                # check that the rate of convergence is steady
                @test (lerr[3]-lerr[1])/(npts[3]-npts[1]) ≈ (lerr[4]-lerr[2])/(npts[4]-npts[2]) atol=1e-2
            end
        end
    end

    @testset "MonkhorstPack" begin
        @testset "volume" begin
            f(x) = 1.0
            buffer = Float64[]
            for npt in (30,), dims in 1:3
                B = rand(dims,dims)
                dom = Basis(B)
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                sols = Vector{Float64}(undef, 2*length(test_syms))
                for (i,syms) in enumerate(test_syms)
                    rule = MonkhorstPack(Float64, Val(dims), npt, syms)
                    nsyms = isnothing(syms) ? 1 : length(syms)
                    sols[2i-1] = rule(f, dom)*nsyms
                    sols[2i] = rule(f, dom, buffer)*nsyms
                end
                @test all(isapprox(abs(det(B))), sols)
            end
        end

        @testset "oscillations" begin
            f(x) = 1.0 + sum(cos, x)
            buffer = Float64[]
            for npt in (30,), dims in 1:3
                B = 2pi*I(dims)
                dom = Basis(B)
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                sols = Vector{Float64}(undef, 2*length(test_syms))
                for (i,syms) in enumerate(test_syms)
                    rule = MonkhorstPack(Float64, Val(dims), npt, syms)
                    sols[2i-1] = rule(f, dom)*length(syms)
                    sols[2i] = rule(f, dom, buffer)*length(syms)
                end
                @test all(isapprox(abs(det(B))), sols)
            end
        end

        @testset "convergence" begin
            f(x, ω=0.0, η=1e-1) = inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
            for dims in 1:2
                dom = Basis(I(dims))
                ref = PTR(Float64, Val(dims), 100)(f, dom)
                npts = (10, 20, 40, 80) # geometric progression for asymptotic behavior
                sols = Vector{ComplexF64}(undef, length(npts))
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                for syms in test_syms
                    for (i,npt) in enumerate(npts)
                        sols[i] = length(syms)*MonkhorstPack(Float64, Val(dims), npt, syms)(f, dom)
                    end
                    @test issorted(reverse(norm.(sols .- ref))) # check errors decrease
                    # error for PTR goes like err_n ~< exp(-aηn)
                    lerr = log10.(norm.(sols .- ref))
                    # check that the rate of convergence is steady
                    @test (lerr[3]-lerr[1])/(npts[3]-npts[1]) ≈ (lerr[4]-lerr[2])/(npts[4]-npts[2]) atol=1e-2
                end
            end
        end


        @testset "symmetry" begin
            f(x, ω=0.0, η=1e-1) =
                inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
            # test symmetries for scalar integrand
            for npt in (30,), dims in 1:3
                dom = Basis(I(dims))
                csym = collect(cube_automorphisms(Val(dims)))
                ref = PTR(Float64, Val(dims), npt)(f, dom)
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                sols = Vector{ComplexF64}(undef, length(test_syms))
                for (i,syms) in enumerate(test_syms)
                    sols[i] = MonkhorstPack(Float64, Val(dims), npt, syms)(f, dom)*length(syms)
                end
                @test all(isapprox(ref), sols)
            end

            # test symmetries for matrix integrand
            # a test matrix with inversion and 4-fold rotation symmetry
            function h(k, t1=0.44, t2=0.13, t3=0.08)
                kx, ky = k
                h11 = -2*t1*(cospi(2kx) + cospi(2ky))
                h21 = t3*sinpi(2kx)*sinpi(2ky)
                h22 = -2*t2*(cospi(2kx) + cospi(2ky))
                SHermitianCompact{2,Float64,3}(SVector{3,Float64}(h11,h21,h22))
            end
            for dims in (2,)
                npt = 30
                dom = Basis(I(dims))
                ref = PTR(Float64, Val(dims), npt)(h, dom)
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                for syms in test_syms
                    int = MonkhorstPack(Float64, Val(dims), npt, syms)(h, dom)*length(syms)
                    int_ = zero(int)
                    for S in syms
                        int_ += S * int * S'
                    end
                    @test norm(ref - int_) < 1e-10
                end
            end
        end

    end

    @testset "pquadrature" begin
        struct HyperCube{d,T}
            a::SVector{d,T}
            b::SVector{d,T}
        end
        endpoints(c::HyperCube) = (c.a, c.b)
        Base.ndims(::HyperCube{d}) where {d} = d
        Base.eltype(::Type{HyperCube{d,T}}) where {d,T} = T

        chebpoints(T, order, dim) = map(n -> T(cospi(n/order)), 0:order)
        # make a new quadrature rule to test (very similar to PTR)
        # based on a tensor product of Clenshaw Curtis rules
        # only works for scalar integrands
        struct ClenshawCurtis{d,T}
            x::NTuple{d,Vector{T}}
        end
        function ClenshawCurtis(::Type{T}, ::Val{d}, order=16) where {T,d}
            pts = chebpoints.(T, order, 1:d)
            return ClenshawCurtis(tuple(pts...))
        end
        Base.length(rule::ClenshawCurtis) = prod(length, rule.x)
        function AutoSymPTR.nextrule(rule::ClenshawCurtis{d,T}, ::Type{ClenshawCurtis}) where {d,T}
            return ClenshawCurtis(T, Val(d), 2 .* length.(rule.x))
        end

        function (rule::ClenshawCurtis{d,T})(f, dom, buffer=nothing) where {d,T}
            a, b = endpoints(dom)
            c = (b - a)/2
            x = Iterators.product(rule.x...)
            vals = map(y -> f(c.*(SVector{d,T}(y)+ones(SVector{d,T}))+a), x)
            kind = map(n -> n > 1 ? FFTW.REDFT00 : FFTW.DHT, size(vals))
            FFTW.r2r!(vals, kind)

            # renormalize the result to obtain the conventional
            # Chebyshev-polnomial coefficients
            s = size(vals)
            vals ./= prod(map(n -> n > 1 ? 2(n-1) : 1, s))
            for dim = 1:d
                # normalization to Chebyshev coefficients
                if size(vals, dim) > 1
                    idx = CartesianIndices(ntuple(i -> i == dim ? (2:s[i]-1) : (1:s[i]), Val{d}()))
                    vals[idx] .*= 2
                end
                # quadrature rule
                preI = CartesianIndices(axes(vals)[1:dim-1])
                sufI = CartesianIndices(axes(vals)[dim+1:d])
                for i in axes(vals, dim)
                    vals[preI, i, sufI] .*=  isodd(i) ? 2/(1 - (i-1)^2) : 0
                end
            end
            return sum(vals) * abs(prod(c))
        end

        @testset "volume" begin
            f(x) = 1.0
            for dims in 1:3
                dom = HyperCube(rand(SVector{dims,Float64}), rand(SVector{dims,Float64}))
                I, = pquadrature(f, dom, ClenshawCurtis)
                @test I ≈ abs(prod(dom.a-dom.b))
            end
        end

        @testset "polynomials" begin

        end

        @testset "convergence" begin

        end
    end

    @testset "autosymptr" begin
        # test bases integrate to correct volume
        for dims in 1:3
            B = Basis(rand(dims,dims))
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = autosymptr(x -> 1, B, syms = syms)[1]*nsyms
            end
            @test all(isapprox(abs(det(B.B))), sols)
        end

        f(x, ω=0.0, η=1.0) =
            inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
        # test symmetries for scalar integrand
        for dims in 1:3
            B = Basis(I(dims))
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = autosymptr(f, B, syms = syms)[1]*nsyms
            end
            @test all(isapprox(sols[1]), sols[2:end])
        end

        # test symmetries for matrix integrand
            # a test matrix with inversion and 4-fold rotation symmetry
            function h(k, t1=0.44, t2=0.13, t3=0.08)
                kx, ky = k
                h11 = -2*t1*(cospi(2kx) + cospi(2ky))
                h21 = t3*sinpi(2kx)*sinpi(2ky)
                h22 = -2*t2*(cospi(2kx) + cospi(2ky))
                SHermitianCompact{2,Float64,3}(SVector{3,Float64}(h11,h21,h22))
            end
            for dims in (2,)
                atol = 1e-5
                dom = Basis(I(dims))
                ref = autosymptr(h, dom; abstol=atol)[1]
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
                for syms in test_syms
                    nsyms = isnothing(syms) ? 1 : length(syms)
                    int = autosymptr(h, dom, syms = syms, abstol=atol/nsyms)[1]
                    int_ = zero(int)
                    for S in syms
                        int_ += S * int * S'
                    end
                    @test norm(ref - int_) < 2atol
                end
            end

        # test convergence threshold achieved within a factor of two
        for dims in 1:2
            g(x, ω=0.0, η=1e-2) = f(x, ω, η)
            dom = Basis(I(dims))
            ref = autosymptr(g, dom; abstol=10^(-10))[1]
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (1.0I,), (-1.0I,1.0I), csym, reverse(csym)) # order of symmetries shouldn't matter
            for atol in (10.0 .^ [-2, -4, -6]), syms in test_syms
                nsyms = isnothing(syms) ? 1 : length(syms)
                @test ref ≈ nsyms*autosymptr(g, dom, syms = syms, abstol=atol/nsyms)[1] atol=2atol
            end
        end

    end

    @testset "BatchIntegrand" begin
        f(x) = 1.0 + sum(cos, x)
        f!(y, x) = y .= f.(x)
        buffer = Float64[]
        for npt in (7,30,), dims in 1:3, max_batch in (1, 2, 3, 4, 5, 29, 30, 31, typemax(Int))
            g = AutoSymPTR.BatchIntegrand(f!, Float64[], SVector{dims,Float64}[], max_batch=max_batch)
            B = 2pi*I(dims)
            dom = Basis(B)
            rule = PTR(Float64, Val(dims), npt)
            @test abs(det(B)) ≈ rule(f, dom) ≈ rule(g, dom) ≈ rule(g, dom, buffer)
        end
    end
end
