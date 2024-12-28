using ConScape, Test, SparseArrays
using Rasters, ArchGDAL, Plots
using LinearSolve

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

mov_prob = replace_missing(Raster(joinpath(datadir, "mov_prob_1000.asc")), NaN)
hab_qual = replace_missing(Raster(joinpath(datadir, "hab_qual_1000.asc")), NaN)
rast = ConScape.coarse_graining(RasterStack((; affinities=mov_prob, qualities=hab_qual)), 10)

graph_measures = graph_measures = (;
    func=ConScape.ConnectedHabitat(),
    qbetw=ConScape.BetweennessQweighted(),
    kbetw=ConScape.BetweennessKweighted(),
)
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor())
connectivity_measure = ConScape.ExpectedCost(θ=1.0)

expected_layers = (:func_exp, :func_oddsfor, :qbetw_exp, :qbetw_oddsfor, :kbetw_exp, :kbetw_oddsfor)

# Basic Problem
problem = ConScape.Problem(; 
    graph_measures, distance_transformation, connectivity_measure,
    solver = ConScape.MatrixSolver(),
)
@btime result = ConScape.solve(problem, rast)
@test result isa RasterStack
@test keys(result) == expected_layers

# Problem with a LinearSolve.jl solver
using BenchmarkTools
linearsolve_problem = ConScape.Problem(; 
    graph_measures, distance_transformation, connectivity_measure,
    solver = ConScape.LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)); threaded=true),
)
@test ls_result isa RasterStack
@test keys(ls_result) == expected_layers

windowed_problem = ConScape.WindowedProblem(problem; 
    radius=40, overlap=10,
)
windowed_result = ConScape.solve(windowed_problem, rast)
@test windowed_result isa RasterStack
@test size(windowed_result) == size(rast)
@test keys(windowed_result) == expected_layers 

stored_problem = ConScape.StoredProblem(problem; 
    path=tempdir(), radius=40, overlap=10, threaded=true
)
ConScape.solve(stored_problem, rast)
stored_result = mosaic(stored_problem; to=rast)
@test stored_result isa RasterStack
@test size(stored_result) == size(rast)
# keys are sorted now from file-name order
@test keys(stored_result) == Tuple(sort(collect(expected_layers)))
# Check the answer matches the WindowedProblem
@test all(stored_result.func_exp .=== windowed_result.func_exp)

# Run a stored problem as batch jobs
stored_problem2 = ConScape.StoredProblem(problem; 
    path=tempdir(), radius=40, overlap=10, threaded=true
)
jobs = ConScape.batch_ids(stored_problem2, rast) 
@test jobs isa Vector{Int}

for job in jobs
    ConScape.solve(stored_problem2, rast, job)
end
bach_result = mosaic(stored_problem2; to=rast)
# Check the answer matches the non-batched run
@test all(bach_result.func_exp .=== stored_result.func_exp)
