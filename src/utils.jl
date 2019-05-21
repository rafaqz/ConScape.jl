# simulate a permeable wall
function perm_wall_sim(
    nrows::Integer,
    ncols::Integer;
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

⊗ = kron
#=
Generate the affinity matrix of a grid graph, where each
pixel is connected to its vertical and horizontal neighbors.

Parameters:
- nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
=#
function _generateA(nrows, ncols, nhood_size)

    if !(nhood_size ∈ (4, 8))
        throw(ArgumentError("nhood_size must be either 4 or 8"))
    end

    A = spdiagm(0=>ones(ncols)) ⊗ spdiagm(-1=>ones(nrows - 1), 1=>ones(nrows - 1)) +
        spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗ spdiagm(0=>ones(nrows))

    if nhood_size == 8
        A .+= spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗
              spdiagm(-1=>fill(1/√2, nrows - 1), 1=>fill(1/√2, nrows - 1))
    end

    return A
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

@enum AdjacencyWeight begin
    TargetWeight
    AverageWeight
end

"""
    adjacency(R::Matrix[, neighbors::Tuple=N8, weight=TargetWeight]) -> SparseMatrixCSC

Compute an adjacency matrix of the raster image `R` of the similarities/conductances
the cells. The similarities are computed as either the value of the target cell (TargetWeight)
or as harmonic means of the cell values weighted by the grid distance (AverageWeight). The similarities
can be computed with respect to eight neighbors (`N8`) or four neighbors (`N4`).
"""
function adjacency(R::Matrix; neighbors::Tuple=N8, weight=TargetWeight)
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
                    if weight == TargetWeight
                        push!(vs, rijk)
                    elseif weight == AverageWeight
                        v = 2/((inv(rij) + inv(rijk))*l)
                        push!(vs, v)
                    else
                        throw(ArgumentError("weight mode not implemented"))
                    end
                end
            end
        end
    end
    return sparse(is, js, vs, m*n, m*n)
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