# This file is a work in progress...

"""
    WindowedProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.
"""
@kwdef struct WindowedProblem{P,R,M} <: AbstractProblem
    problem::P
    ranges::R
    mask::M
end
function WindowedProblem(problem; 
    target=nothing, 
    radius,
    overlap,
    res=resolution(target),
)
    ranges = _get_ranges(res, radius, overlap)
    mask = _get_mask(target, ranges)
    WindowedProblem(problem, ranges, mask)
end

function compute(p::WindowedProblem, rast::RasterStack)
    outputs = map(p.mask, p.ranges) do m, r
        m || return missing
        target = rast[r...]
        p1 = _initialise(p, target)
        g = Grid(p1, target)
        compute(p, g)
    end
    # Return mosaics of outputs
    return _mosaic_by_measure(p, outputs)
end

_mosaic_by_measure(p::WindowedProblem, outputs) = _mosaic_measures(p.problem, outputs)
_mosaic_by_measure(p::Problem, outputs) = _mosaic_measures(p.graph_measures,p, outputs)
function _mosaic_by_measure(gm::NamedTuple{K}, p::WindowedProblem, outputs) where K
    map(K) do k
        to_mosaic = map(outputs) do o
            o[k]
        end
        Rasters.mosaic(sum, to_mosaic)
    end |> NamedTuple{K}
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
struct StoredProblem{P,R,M}
    problem::P
    ranges::R
    mask::M
    path::String
    ext::String
end
function StoredProblem(p::AbstractProblem;
    target::Raster,
    radius::Real,
    overlap::Real,
    path::String,
    ext::String=".tif",
)
    res = resolution(target)
    ranges = _get_ranges(res, radius, overlap)
    mask = _get_mask(target, ranges)
    return StoredProblem(w, ranges, mask, path, ext)
end

function compute(p::StoredProblem, rast::RasterStack)
    map(p.ranges, p.mask) do rs, m
        m || return nothing
        target = rast[r...]
        p1 = _initialise(p, target)
        g = Grid(p1, target)
        output = compute(p, g)
        _store(p, output, rs)
        nothing
    end
end

function _store(p::StoredProblem, output::NamedTuple{K}, ranges) where K
    window_path = mkpath(_window_path(p, ranges))
    map(K) do k
        filepath = joinpath(window_path, string(k) * p.file_extension)
        Rasters.write(filepath, output[k])
    end
end

function _window_path(p, ranges)
    corners = map(first, ranges)
    window_dirname =  "window_" * join(corners, '_')
    return joinpath(p.path, window_dirname)
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(p::StoredProblem; to=nothing, filename=nothing)
    window_paths = [_window_path(p, r) for (rs, m) in zip(p.ranges, p.mask) if m]
    K = keys(graph_measures(p))
    rasters = map(K) do name
        filepaths = joinpath.(window_paths, (string(name) * p.file_extension,))
        windows = [Raster(fp; lazy=true, name) for fp in filepaths if isfile(fp)]
        # Mosaic them all together
        filename = filename * '_' * name * p.file_extension
        return Rasters.mosaic(sum, windows; to, filename)
    end |> NamedTuple{K}

    return RasterStack(rasters)
end

# Generate a new mask if nested
_initialise(p::Problem, target) = p
function _initialise(p::WindowedProblem, target)
    mask = _get_mask(target, p.ranges)
    WindowedProblem(p.problem, p.ranges, mask)
end
function _initialise(p::StoredProblem, target)
    mask = _get_mask(target, p.ranges)
    StoredProblem(p.problem, p.ranges, mask, p.path)
end

function _get_ranges(res, radius, overlap)
    # Convert distances to pixels
    r = floor(Int, radius / res)
    o = floor(Int, overlap / res)
    s = r - o # Step between each window corner
    # Define the corners of each window
    corners = CartesianIndices(target)[begin:s:end, begin:s:end]
    # Create an array of ranges for retreiving each window
    ranges = map(corners) do corner
        map(corner, size(target)) do i, sz
            i:min(sz, i + r)
        end
    end
    return ranges
end

_get_mask(target::Nothing, ranges) = nothing
function _generate_mask(target, ranges)
    # Create a mask to skip tiles that have no target cells
    map(ranges) do I
        # Get a window view
        window = view(target, I...)
        # If there are non-NaN cells above zero, keep the window
        # TODO allow users to change this condition?
        any(x -> !isnan(x) && x > zero(x), window)
    end
end