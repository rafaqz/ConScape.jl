module ConScape

    using SparseArrays, Plots, LightGraphs, SimpleWeightedGraphs
    using LinearAlgebra

    # Special matrix for efficient inverse
    include("blocktridiagonal.jl")

    mutable struct Grid
        nrows::Int
        ncols::Int
        A::SparseMatrixCSC{Float64,Int}
        id_to_grid_coordinate_list::Vector{Tuple{Int,Int}}
        source_qualities::Matrix{Float64}
        target_qualities::Matrix{Float64}
    end

    """
        Grid(nrows::Integer, ncols::Integer;
                  qualities::Matrix=ones(nrows, ncols),
                  source_qualities::Matrix=qualities,
                  target_qualities::Matrix=qualities,
                  nhood_size::Integer=8,
                  landscape=_generateA(nrows, ncols, nhood_size)) -> Grid

    Construct a `Grid` from a `landscape` passed a `SparseMatrixCSC`.
    """
    function Grid(nrows::Integer, ncols::Integer;
                  qualities::Matrix=ones(nrows, ncols),
                  source_qualities::Matrix=qualities,
                  target_qualities::Matrix=qualities,
                  nhood_size::Integer=8,
                  landscape=_generateA(nrows, ncols, nhood_size))
        @assert nrows*ncols == LinearAlgebra.checksquare(landscape)
        Ngrid = nrows*ncols

        Grid(nrows,
             ncols,
             landscape,
             _id_to_grid_coordinate_list(Ngrid, ncols),
             source_qualities,
             target_qualities)
    end

    Base.size(g::Grid) = (g.nrows, g.ncols)

    # simulate a permeable wall
    function perm_wall_sim(nrows::Integer, ncols::Integer;
                           scaling::Float64=0.5,
                           wallwidth::Integer=3,
                           wallposition::Float64=0.5,
                           corridorwidths::NTuple{<:Any,<:Integer}=(3,3),
                           corridorpositions=(0.35,0.7),
                           kwargs...)

        # 1. initialize landscape
        N = nrows*ncols
        g = Grid(nrows, ncols; kwargs...)
        g.A = scaling * g.A

        # # 2. compute the wall
        wpt = round(Int, ncols*wallposition - wallwidth/2 + 1)
        xs  = range(wpt, stop=wpt + wallwidth - 1)

        # 3. compute the corridors
        ys = Int[]
        for i in 1:length(corridorwidths)
            cpt = floor(Int, nrows*corridorpositions[i]) - ceil(Int, corridorwidths[i]/2)
            if i == 1
                append!(ys, 1:cpt)
            else
                append!(ys, range(maximum(ys) + 1 + corridorwidths[i-1], stop=cpt))
                append!(ys, range(maximum(ys) + 1 + corridorwidths[i]  , stop=nrows))
            end
        end

        impossible_nodes = vec(collect(Iterators.product(ys,xs)))
        _set_impossible_nodes!(g, impossible_nodes)

        return g
    end

    #=
    Generate the affinity matrix of a grid graph, where each
    pixel is connected to its vertical and horizontal neighbors.

    Parameters:
    - nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
    =#
    function _generateA(nrows, ncols, nhood_size)

        N = nrows*ncols

        # A = ss.dok_matrix(N, N)
        is, js, vs = Int[], Int[], Float64[]
        for i in 1:nrows
            for j in 1:ncols
                n = (i - 1)*ncols + j # current pixel
                if j < ncols
                    # Add horizontal edge:
                    # A[n, n + 1] = 1
                    push!(is, n)
                    push!(js, n + 1)
                    push!(vs, 1)
                end
                if i < nrows
                    # Add vertical edge:
                    # A[n, n + ncols] = 1
                    push!(is, n)
                    push!(js, n + ncols)
                    push!(vs, 1)

                    # TODO: WRITE THIS TO ALLOW OTHER VALUES OF nhood_size!
                    if nhood_size == 8
                        if j < ncols
                            # Add lower-right diagonal edge:
                            # A[n, n + ncols + 1] = 1 / √2
                            push!(is, n)
                            push!(js, n + ncols + 1)
                            push!(vs, 1 / √2)
                        end
                        if j > 1
                            # Add lower-left diagonal edge:
                            # A[n, n+ncols-1] = 1 / √2
                            push!(is, n)
                            push!(js, n + ncols - 1)
                            push!(vs, 1 / √2)
                        end
                    end
                end
            end
        end

        A = sparse(is, js, vs, N, N)

        return A + A'         # Symmetrize
    end

    const N4 = (( 0, -1, 1.0),
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                ( 0,  1, 1.0))
    const N8 = ((-1, -1,  √2),
                ( 0, -1, 1.0),
                ( 1, -1,  √2),
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                (-1,  1,  √2),
                ( 0,  1, 1.0),
                ( 1,  1,  √2))

    """
        adjacency(R::Matrix[, neighbors::Tuple=N8]) -> SparseMatrixCSC

    Compute an adjacency matrix of the raster image `R` of the similarities/conductances
    the cells. The similarities are computed as harmonic means of the cell values weighted
    by the grid distance. The similarities can be computed with respect to eight
    neighbors (`N8`) or four neighbors (`N4`).
    """
    function adjacency(R::Matrix, neighbors::Tuple=N8)
        m, n = size(R)

        # Initialy the buffers of the SparseMatrixCSC
        is, js, vs = Int[], Int[], Float64[]

        for j in 1:n
            for i in 1:m
                # Base node
                rij = R[i, j]
                for (ki, kj, l) in neighbors
                    if !(1 <= i + ki <= m) || !(1 <= j + kj <= n)
                        # Continue when computing edge out of raster image
                        continue
                    else
                        # Target node
                        rijk = R[i + ki, j + kj]
                        if iszero(rijk)
                            # Don't include zero similaritiers
                            continue
                        end

                        push!(is, (j - 1)*m + i)
                        push!(js, (j - 1)*m + i + ki + kj*m)
                        v = 2/((inv(rij) + inv(rijk))*l)
                        push!(vs, v)
                    end
                end
            end
        end
        return sparse(is, js, vs)
    end

    function _id_to_grid_coordinate_list(N_grid, ncols)
        id_to_grid_coordinate_list = Tuple{Int,Int}[]
        for node_id in 1:N_grid
            j = (node_id - 1) % ncols + 1
            i = div(node_id - j, ncols) + 1
            push!(id_to_grid_coordinate_list, (i,j))
        end
        return id_to_grid_coordinate_list
    end

    #=
    Make pixels impossible to move to by changing the affinities to them to zero.
    Input:
        - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
    =#
    function _set_impossible_nodes!(g::Grid, node_list::Vector{<:Tuple}, impossible_affinity=1e-20)
        # Find the indices of the coordinates in the id_to_grid_coordinate_list vector
        node_list_idx = [findfirst(isequal(n), g.id_to_grid_coordinate_list) for n in node_list]

        A = g.A

        # Set (nonzero) values to impossible_affinity:
        if impossible_affinity > 0
            A[node_list_idx,:] = impossible_affinity*(A[node_list_idx,:] .> 0)
            A[:,node_list_idx] = impossible_affinity*(A[:,node_list_idx] .> 0)
        elseif impossible_affinity == 0
            # Delete the nodes completely:
            num_of_removed = length(node_list_idx)

            nodes_to_keep = [n for n in 1:size(A, 1) if !(n in node_list_idx)]

            A = A[nodes_to_keep,:]
            A = A[:,nodes_to_keep]

            # FIXME! Commented out 8 April 2019 since qualities are now matrices.
            # Check if commention out is a problem. I don't think so.
            # deleteat!(vec(g.source_qualities), node_list_idx)
            # deleteat!(vec(g.target_qualities), node_list_idx)
            g.id_to_grid_coordinate_list = [g.id_to_grid_coordinate_list[id] for id in 1:length(g.id_to_grid_coordinate_list) if !(id in node_list_idx)]
        end

        g.A = A
    end

    """
        mapnz(f, A::SparseMatrixCSC) -> SparseMatrixCSC

    Map the non-zero values of a sparse matrix `A` with the function `f`.
    """
    function mapnz(f, A::SparseMatrixCSC)
        B = copy(A)
        map!(f, B.nzval, A.nzval)
        return B
    end

    function plot_outdegrees(g::Grid)
        values = sum(g.A, dims=2)
        canvas = zeros(g.nrows, g.ncols)
        for (i,v) in enumerate(values)
            canvas[g.id_to_grid_coordinate_list[i]...] = v
        end
        heatmap(canvas)
    end

    abstract type Cost end
    struct MinusLog <: Cost end
    struct ExpMinus <: Cost end
    struct Inv      <: Cost end

    (::MinusLog)(x::Number) = -log(x)
    (::ExpMinus)(x::Number) = exp(-x)
    (::Inv)(x::Number)      = inv(x)

    Base.inv(::MinusLog) = ExpMinus()
    Base.inv(::ExpMinus) = MinusLog()
    Base.inv(::Inv)      = Inv()

    struct Habitat
        g::Grid
        cost::Cost
        C::SparseMatrixCSC{Float64,Int}
        Pref::SparseMatrixCSC{Float64,Int}
        # landmarks::Vector{Int}
    end

    """
        Habitat(g::Grid, cost::Cost) -> Habitat

    Construct a Habitat from a `g::Grid` based on a `costfunction`.
    """
    Habitat(g::Grid,
            cost::Cost) =
                Habitat(g,
                        cost,
                        mapnz(cost, g.A),
                        _Pref(g.A))

    _Pref(A::SparseMatrixCSC) = sum(A, dims=2) .\ A

    function _W(Pref::SparseMatrixCSC, β::Real, C::SparseMatrixCSC)

        n = LinearAlgebra.checksquare(Pref)
        if LinearAlgebra.checksquare(C) != n
            throw(DimensionMismatch("Pref and C must have same size"))
        end

        return Pref .* exp.((-).(β) .* C)
    end

    _W(h::Habitat; β=nothing) = _W(h.Pref, β, h.C)

    """
        RSP_full_betweenness_qweighted(h::Habitat) -> Matrix

    Compute full RSP betweenness of all nodes weighted by source and target qualities.
    """
    RSP_full_betweenness_qweighted(h::Habitat; β=nothing) =
        RSP_full_betweenness_qweighted(inv(Matrix(I - _W(h, β=β))), h.g.source_qualities, h.g.target_qualities)

    function RSP_full_betweenness_qweighted(Z::AbstractMatrix,
                                            source_qualities::AbstractMatrix,
                                            target_qualities::AbstractMatrix)

        Zdiv = inv.(Z)
        Zdiv_diag = diag(Zdiv)

        qs = vec(source_qualities)
        qt = vec(target_qualities)
        qs_sum = sum(qs)

        ZQZdivQ = qt .* Zdiv'
        ZQZdivQ = ZQZdivQ .* qs'
        ZQZdivQ -= Diagonal(qs_sum .* qt .* Zdiv_diag)

        ZQZdivQ = Z*ZQZdivQ

        return reshape(sum(ZQZdivQ .* Z', dims=2), reverse(size(source_qualities))...)'
    end

    """
        RSP_full_betweenness_kweighted(h::Habitat) -> Matrix

    Compute full RSP betweenness of all nodes weighted with proximity.
    """
    function RSP_full_betweenness_kweighted(h::Habitat; β=nothing)
        W = _W(h, β=β)
        Z = inv(Matrix(I - W))
        similarities = map(inv(h.cost), RSP_dissimilarities(W, h.C, Z))
        similarities[diagind(similarities)] .= 0
        return RSP_full_betweenness_kweighted(Z, h.g.source_qualities, h.g.target_qualities, similarities)
    end

    function RSP_full_betweenness_kweighted(Z::AbstractMatrix,
                                            source_qualities::AbstractMatrix,
                                            target_qualities::AbstractMatrix,
                                            similarities::AbstractMatrix)

        Zdiv = inv.(Z)

        qs = vec(source_qualities)
        qt = vec(target_qualities)

        K = qs .* similarities .* qt'

        K_colsum = vec(sum(K, dims=1))
        d_Zdiv = diag(Zdiv)

        ZKZdiv = K .* Zdiv
        ZKZdiv -= Diagonal(K_colsum .* d_Zdiv)

        ZKZdiv = Z*ZKZdiv
        bet = sum(ZKZdiv .* Z', dims=1)

        return reshape(bet, reverse(size(source_qualities)))'
    end

    """
        RSP_dissimilarities(h::Habitat) -> Matrix
    Compute RSP expected costs or RSP dissimilarities from all nodes
    """
    RSP_dissimilarities(h::Habitat; β=nothing) = RSP_dissimilarities(_W(h, β=β), h.C)

    function RSP_dissimilarities(W::SparseMatrixCSC, C::SparseMatrixCSC, Z::AbstractMatrix = inv(Matrix(I - W)))
        n   = LinearAlgebra.checksquare(W)
        CW  = C .* W
        S   = (Z*(C .* W)*Z) ./ Z
        d_s = diag(S)
        C̄   = S .- d_s
        return C̄
    end
end
