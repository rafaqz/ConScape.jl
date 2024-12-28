# Defined in ConScape.jl for load order
# abstract type Solver end
@doc """
    Solver

Abstract supertype for ConScape solvers.
""" Solver 

# RSP is not used for ConnectivityMeasure, so the solver isn't used
function solve(::Solver, cm::ConnectivityMeasure, p::AbstractProblem, g::Grid) 
    return map(p.graph_measures) do gm
        compute(gm, p, g)
    end
end

"""
   MatrixSolver()

Solve all operations on a fully materialised Z matrix.

This is fast but memory inneficient for CPUS, and isn't threaded.
But may be best for GPUs using CuSSP.jl ?
"""
@kwdef struct MatrixSolver <: Solver 
    check::Bool = true
end

# Materialise the whole rhs matrix
function solve(s::MatrixSolver, cm::FundamentalMeasure, p::AbstractProblem, g::Grid) 
    (; A, B, Pref, W) = setup_sparse_problem(g, cm)
    # Nearly all the work and allocation happens here
    Z = A \ Matrix(B)
    # Check that values in Z are not too small:
    if s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    results = map(p.graph_measures) do gm
        compute(gm, p, grsp)
    end
    return _merge_to_stack(results)
end

"""
   LinearSolver(args...; threded, kw...)

Solve all operations column-by-column using LinearSolve.jl solvers.

The `threaded` keyword specifies if threads are used per target.
Other arguments and keywords are passed to `LinearSolve.solve` after the
problem object, like:

````julia
`LinearSolve.solve(linearproblem, args...; kw...)`
````

# Example

This example uses LinearSolve.jl wth `KrylovJL_GMRES` and a preconditioner.

TODO: an example that is realistic

````julia
using LinearSolve
problem = ConScape.Problem(; 
    solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
    graph_measures = (;
        func=ConScape.ConnectedHabitat(),
        qbetw=ConScape.BetweennessQweighted(),
    ),
    distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor()),
    connectivity_measure = ConScape.ExpectedCost(θ=1.0),
)
````
"""
struct LinearSolver <: Solver 
    args
    keywords
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

# Use an iterative solver so the grid is not materialised
function solve(s::LinearSolver, cm::FundamentalMeasure, p::AbstractProblem, g::Grid)
    (; A, B) = setup_sparse_problem(g, cm)
    # Dense rhs column
    b_init = zeros(eltype(A), size(B, 1))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b_init)
    linsolve = init(linprob, s.args...; s.keywords...)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    Z = Matrix{eltype(A)}(undef, size(B))
    if s.threaded
        nbuffers = Threads.nthreads()
        # Create a channel to store problem b vectors for threads
        # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
        ch = Channel{Tuple{typeof(linsolve),Vector{Float64}}}(nbuffers)
        for i in 1:nbuffers
            # TODO fix this in LinearSolve.jl with batching
            # We should not need to `deepcopy` the whole problem we 
            # just need to replicate the specific workspace arrays 
            # that will cause race conditions.
            # But currently there is no parallel mode for LinearSolve.jl
            # See https://github.com/SciML/LinearSolve.jl/issues/552
            put!(ch, (deepcopy(linsolve), Vector{eltype(A)}(undef, size(B, 1))))
        end
        Threads.@threads for i in 1:size(B, 2)
            # Get column memory from the channel
            linsolve_t, b_t = take!(ch)
            # Update it
            b_t .= view(B, :, i)
            # Update solver with new b values
            reinit!(linsolve_t; b=b_t, reuse_precs=true)
            sol = LinearSolve.solve(linsolve_t, s.args...; s.keywords...)
            # Aim for something like this ?
            # res = map(connectivity_measures(p)) do cm
            #     compute(cm, g, sol.u, i)
            # end
            # For now just use Z
            Z[:, i] .= sol.u
            put!(ch, (linsolve_t, b_t))
        end
    else
        for i in 1:size(B, 2)
            b_init .= view(B, :, i)
            reinit!(linsolve; b=b_init, reuse_precs=true)
            sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
            # Udate the column
            Z[:, i] .= sol.u
        end
    end
    # return _combine(res, g) # return results as Rasters

    # TODO remove all use of GridRSP
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    results = map(p.graph_measures) do gm
        compute(gm, p, grsp)
    end 
    return _merge_to_stack(results)
end


# Utils

function setup_sparse_problem(g::Grid, cm::FundamentalMeasure)
    Pref = _Pref(g.affinities)
    W = _W(Pref, cm.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse diagonal rhs matrix
    B = sparse(g.targetnodes,
        1:length(g.targetnodes),
        1.0,
        size(g.costmatrix, 1),
        length(g.targetnodes),
    )
    return (; A, B, Pref, W)
end

# We may have multiple distance_measures per
# graph_measure, but we want a single RasterStack.
# So we merge the names of the two layers
function _merge_to_stack(nt::NamedTuple{K}) where K
    unique_nts = map(K) do k
        gm = nt[k]
        if gm isa NamedTuple
            # Combine outer and inner names with an underscore
            joinedkeys = map(keys(gm)) do k_inner
                Symbol(k, :_, k_inner)
            end
            # And rename the NamedTuple
            NamedTuple{joinedkeys}(values(gm))
        else
            # We keep the name as is
            NamedTuple{(k,)}((gm,))
        end
    end
    # merge unique layers into a sinlge RasterStack
    return RasterStack(merge(unique_nts...))
end