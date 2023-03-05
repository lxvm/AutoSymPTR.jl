using LinearAlgebra
using Test

using StaticArrays

using AutoSymPTR

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

    @testset "symptr" begin

        #= test keywords
        for npt in (10,), dim in 1:3, syms in (nothing, (I,))
            int, rule = symptr(x -> 1, I(dim), syms) # default settings
            @test int ≈ symptr(x -> 1, I(dim), syms; rule...)[1]
            int, rule = symptr(x -> 1, I(dim), syms; npt=npt) # user settings
            @test int ≈ symptr(x -> 1, I(dim), syms; npt=npt, rule...)[1] # uses precomputed rule instead of specified npt
        end
        =#

        # test bases integrate to correct volume
        for npt in (30,), dims in 1:3
            B = rand(dims,dims)
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = symptr(x -> 1, B, syms; npt=npt)*nsyms
            end
            @test all(isapprox(abs(det(B))), sols)
        end

        f(x, ω=0.0, η=1e-1) =
            inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
        # test symmetries for scalar integrand
        for npt in (30,), dims in 1:3
            B = I(dims)
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = symptr(f, B, syms; npt=npt)*nsyms
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
            npt = 30
            ref = ptr(h, I(dims); npt=npt)
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = ((I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            for syms in test_syms
                int = symptr(h, I(dims), syms; npt=npt)
                int_ = zero(int)
                for S in syms
                    int_ += S * int * S'
                end
                @test norm(ref - int_) < 1e-10
            end
        end

        # test spectral convergence
        for dims in 1:2
            ref = ptr(f, I(dims); npt=100)
            npts = (10, 20, 40, 80) # geometric progression for asymptotic behavior
            sols = Vector{ComplexF64}(undef, length(npts))
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            for syms in test_syms
                for (i,npt) in enumerate(npts)
                    nsyms = isnothing(syms) ? 1 : length(syms)
                    sols[i] = nsyms*symptr(f, I(dims), syms; npt=npt)
                end
                @test issorted(reverse(norm.(sols .- ref))) # check errors decrease
                # error for PTR goes like err_n ~< exp(-aηn)
                lerr = log10.(norm.(sols .- ref))
                 # check that the rate of convergence is steady
                @test (lerr[3]-lerr[1])/(npts[3]-npts[1]) ≈ (lerr[4]-lerr[2])/(npts[4]-npts[2]) atol=1e-2
            end

        end

    end

    @testset "autosymptr" begin
        #= test keywords
        for dim in 1:3, syms in (nothing, (I,))
            int, err, nevals, rule = autosymptr(x -> 1, I(dim), syms) # default settings
            @test int ≈ autosymptr(x -> 1, I(dim), syms; rule...)[1]
            int, err, nevals, rule = autosymptr(x -> 1, I(dim), syms; atol=1e-1, rtol=0, maxevals=10^7) # user settings
            @test int ≈ autosymptr(x -> 1, I(dim), syms; rule...)[1]
        end
        =#

        # test bases integrate to correct volume
        for dims in 1:3
            B = rand(dims,dims)
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = autosymptr(x -> 1, B, syms)[1]*nsyms
            end
            @test all(isapprox(abs(det(B))), sols)
        end

        f(x, ω=0.0, η=1.0) =
            inv(complex(ω-mapreduce(y -> cospi(2y), +, x), η))
        # test symmetries for scalar integrand
        for dims in 1:3
            B = I(dims)
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            sols = Vector{ComplexF64}(undef, length(test_syms))
            for (i,syms) in enumerate(test_syms)
                nsyms = isnothing(syms) ? 1 : length(syms)
                sols[i] = autosymptr(f, B, syms)[1]*nsyms
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
                ref = autoptr(h, I(dims); atol=atol)[1]
                csym = collect(cube_automorphisms(Val(dims)))
                test_syms = ((I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
                for syms in test_syms
                    int = autosymptr(h, I(dims), syms; atol=atol)[1]
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
            ref = autoptr(g, I(dims); atol=10^(-10))[1]
            csym = collect(cube_automorphisms(Val(dims)))
            test_syms = (nothing, (I,), (-I,I), csym, reverse(csym)) # order of symmetries shouldn't matter
            for atol in (10.0 .^ [-2, -4, -6]), syms in test_syms
                nsyms = isnothing(syms) ? 1 : length(syms)
                @test ref ≈ nsyms*autosymptr(g, I(dims), syms; atol=atol)[1] atol=2atol
            end
        end
        
    end

end