# This file is a work in progress...

"""
    WindowedProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.
"""
struct WindowedProblem{P} <: AbstractProblem
    problem::P
    radius::Int
    overlap::Int
end
WindowedProblem(problem; radius, overlap) =
    WindowedProblem(problem, radius, overlap)

function solve(wp::WindowedProblem, rast::RasterStack)
    ranges = _get_ranges(wp, rast)
    mask = _get_mask(rast, ranges)
    p = wp.problem
    output_stacks = [solve(p, rast[rs...]) for (m, rs) in zip(mask, ranges) if m]
    # Return mosaics of outputs
    return Rasters.mosaic(sum, output_stacks; to=rast)
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
    StoredProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over tiles of windowed grids.
"""
struct StoredProblem{P}
    problem::P
    radius::Int
    overlap::Int
    path::String
    ext::String
end
function StoredProblem(p::AbstractProblem;
    radius::Int,
    overlap::Int,
    path::String,
    ext::String=".tif",
)
    return StoredProblem(p, radius, overlap, path, ext)
end

function solve(sp::StoredProblem, rast::RasterStack)
    ranges = _get_ranges(sp, rast)
    mask = _get_mask(rast, ranges)
    for (rs, m) in zip(ranges, mask)
        m || continue
        output = solve(sp.problem, rast[rs...])
        _store(sp, output, rs)
    end
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(sp::StoredProblem; 
    to, lazy=false, filename=nothing, kw...
)
    ranges = _get_ranges(sp, to)
    mask = _get_mask(to, ranges)
    paths = [_window_path(sp, rs) for (rs, m) in zip(ranges, mask) if m]
    stacks = [RasterStack(p; lazy, name) for p in paths if isdir(p)]

    return Rasters.mosaic(sum, stacks; to, filename, kw...)
end

function _store(p::StoredProblem, output::RasterStack{K}, ranges) where K
    path = mkpath(_window_path(p, ranges))
    Rasters.write(joinpath(path, ""), output; 
        ext=p.ext, verbose=false, force=true
    )
end

function _window_path(p, ranges)
    corners = map(first, ranges)
    window_dirname =  "window_" * join(corners, '_')
    return joinpath(p.path, window_dirname)
end

# Generate a new mask if nested
_initialise(p::Problem, target) = p
function _initialise(p::WindowedProblem, target)
    WindowedProblem(p.problem, p.ranges, mask)
end
function _initialise(p::StoredProblem, target)
    mask = _get_mask(target, p.ranges)
    StoredProblem(p.problem, p.ranges, mask, p.path)
end

_get_ranges(p::Union{StoredProblem,WindowedProblem}, rast::AbstractRasterStack) = 
    _get_ranges(size(rast), p.radius, p.overlap)
function _get_ranges(size::Tuple{Int,Int}, r::Int, overlap::Int)
    r <= overlap && throw(ArgumentError("radius must be larger than overlap"))
    s = r - overlap # Step between each window corner
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:s:end, begin:s:end]
    # Create an iterator of ranges for retreiving each window
    return (map((i, sz) -> i:min(sz, i + r), Tuple(c), size) for c in corners)
end

_get_mask(::Nothing, ranges) = nothing
_get_mask(rast::AbstractRasterStack, ranges) =
    _get_mask(_get_target(rast), ranges)
function _get_mask(target::AbstractRaster, ranges)
    # Create a mask to skip tiles that have no target cells
    map(ranges) do I
        # Get a window view
        window = view(target, I...)
        # If there are non-NaN cells above zero, keep the window
        # TODO allow users to change this condition?
        any(x -> !isnan(x) && x > zero(x), window)
    end
end

resolution(rast) = abs(step(lookup(rast, X)))