# This file is a work in progress...

"""
    WindowedProblem(problem::AbstractProblem; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.

`problem` is usually a [`Problem`](@ref) object but can be any `AbstractProblem`.

# Keywords

- `problem`: The radius of the window.
- `radius`: The radius of the window.
- `overlap`: The overlap between windows.
- `threaded`: Whether to run in parallel. `false` by default
"""
@kwdef struct WindowedProblem <: AbstractProblem
    problem::AbstractProblem
    radius::Int
    overlap::Int
    threaded::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

function solve(wp::WindowedProblem, rast::RasterStack)
    ranges = collect(_get_window_ranges(wp, rast))
    mask = _get_window_mask(rast, ranges)
    p = wp.problem
    output_stacks = Vector{RasterStack}(undef, count(mask))
    used_ranges = ranges[mask]
    if wp.threaded
        Threads.@threads for i in eachindex(used_ranges)
            output_stacks[i] = solve(p, rast[used_ranges[i]...])
        end
    else
        for i in eachindex(used_ranges)
            output_stacks[i] = solve(p, rast[used_ranges[i]...])
        end
    end
    # Return mosaics of outputs
    return Rasters.mosaic(sum, output_stacks; to=rast, missingval=NaN)
end

# function assess(op::WindowedProblem, g::Grid) 
#     window_assessments = map(_windows(op, g)) do w
#         ca = assess(op.op, w)
#     end
#     maximums = reduce(window_assessments) do acc, a
#         (; totalmem=max(acc.totalmem, a.totalmem),
#            zmax=max(acc.zmax, a.zmax),
#            lumax=max(acc.lumax, a.lumax),
#         )
#     end
#     ComputeAssesment(; op=op.op, maximums..., sums...)
# end


"""
    StoredProblem(problem::AbstractProblem; radius, overlap, path, ext)

Combine multiple compute operations into a single object, 
when compute times are long and intermediate storage is needed.

`problem` is usually a [`Problem`](@ref) object or a `WindowedProblem` 
for nested operations.

# Keywords

- `radius`: The radius of the window - 2radius + 1 is the diameter.
- `overlap`: The overlap between adjacent windows.
- `path`: The path to store the output rasters.
- `ext`: The file extension for Rasters.jl to write to. Defaults to `.tif`,
    But can be `.nc` for NetCDF or most other common extensions.
- `threaded`: Whether to run in parallel. `false` by default
"""
@kwdef struct StoredProblem
    problem::AbstractProblem
    radius::Int
    overlap::Int
    path::String
    ext::String = ".tif"
    threaded::Bool = false
end
StoredProblem(problem; kw...) =  StoredProblem(; problem, kw...)

function solve(sp::StoredProblem, rast::RasterStack)
    ranges = collect(_get_window_ranges(sp, rast))
    mask = _get_window_mask(rast, ranges)
    if sp.threaded
        Threads.@threads for rs in ranges[mask]
            output = solve(sp.problem, rast[rs...])
            _store(sp, output, rs)
        end
    else
        for rs in ranges[mask]
            output = solve(sp.problem, rast[rs...])
            _store(sp, output, rs)
        end
    end
end
# Single batch job for clusters
function solve(sp::StoredProblem, rast::RasterStack, i::Int)
    ranges = collect(_get_window_ranges(sp, rast))
    rs = ranges[i]
    output = solve(sp.problem, rast[rs...])
    _store(sp, output, rs)
end

"""
    batch_ids(sp::StoredProblem, rast::RasterStack)

Return the batch indices of the windows that need to be computed.

Returns a `Vector{Int}`
"""
function batch_ids(sp::StoredProblem, rast::RasterStack)
    ranges = _get_window_ranges(sp, rast)
    mask = _get_window_mask(rast, ranges)
    return eachindex(mask)[vec(mask)]
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(sp::StoredProblem; 
    to, lazy=false, filename=nothing, missingval=NaN, kw...
)
    ranges = _get_window_ranges(sp, to)
    mask = _get_window_mask(to, ranges)
    paths = [_window_path(sp, rs) for (rs, m) in zip(ranges, mask) if m]
    stacks = [RasterStack(p; lazy, name) for p in paths if isdir(p)]

    return Rasters.mosaic(sum, stacks; to, filename, missingval, kw...)
end

function _store(p::StoredProblem, output::RasterStack{K}, ranges) where K
    path = mkpath(_window_path(p, ranges))
    return Rasters.write(joinpath(path, ""), output; 
        ext=p.ext, verbose=false, force=true
    )
end

function _window_path(p, ranges)
    corners = map(first, ranges)
    window_dirname =  "window_" * join(corners, '_')
    return joinpath(p.path, window_dirname)
end


### Shared utilities

# Generate a new mask if nested
_initialise(p::Problem, target) = p
function _initialise(p::WindowedProblem, target)
    WindowedProblem(p.problem, p.ranges, mask)
end
function _initialise(p::StoredProblem, target)
    mask = _get_window_mask(target, p.ranges)
    StoredProblem(p.problem, p.ranges, mask, p.path)
end

_get_window_ranges(p::Union{StoredProblem,WindowedProblem}, rast::AbstractRasterStack) = 
    _get_window_ranges(size(rast), p.radius, p.overlap)
function _get_window_ranges(size::Tuple{Int,Int}, r::Int, overlap::Int)
    d = 2r
    d <= overlap && throw(ArgumentError("2radius must be larger than overlap"))
    s = d - overlap # Step between each window corner
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:s:end, begin:s:end]
    # Create an iterator of ranges for retreiving each window
    return (map((i, sz) -> i:min(sz, i + d), Tuple(c), size) for c in corners)
end

_get_window_mask(::Nothing, ranges) = nothing
_get_window_mask(rast::AbstractRasterStack, ranges) =
    _get_window_mask(_get_target(rast), ranges)
function _get_window_mask(target::AbstractRaster, ranges)
    # Create a mask to skip tiles that have no target cells
    map(r -> _has_values(target, r), ranges)
end

function _has_values(target::AbstractRaster, rs::Tuple{Vararg{AbstractUnitRange}})
    # Get a window view
    window = view(target, rs...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    any(x -> !isnan(x) && x > zero(x), window)
end

_resolution(rast) = abs(step(lookup(rast, X)))