"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure end

keywords(o::GraphMeasure) = _keywords(o)

abstract type TopologicalMeasure <: GraphMeasure end
abstract type BetweennessMeasure <: GraphMeasure end
abstract type PerturbationMeasure <: GraphMeasure end
abstract type PathDistributionMeasure <: GraphMeasure end

struct BetweennessQweighted <: BetweennessMeasure end
@kwdef struct BetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end
struct EdgeBetweennessQweighted <: BetweennessMeasure end
@kwdef struct EdgeBetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end

@kwdef struct ConnectedHabitat{DV} <: GraphMeasure
    diagvalue::DV=nothing
end

@kwdef struct Criticality{DV,AV,QT,QS} <: PerturbationMeasure 
    diagvalue::DV=nothing
    avalue::AV=floatmin()
    qˢvalue::QS=0.0
    qᵗvalue::QT=0.0
end

# These maybe don't quite belong here?
@kwdef struct EigMax{F,DV,T} <: TopologicalMeasure
    diagvalue::DV=nothing
    tol::T=1e-14
end

struct MeanLeastCostKullbackLeiblerDivergence <: PathDistributionMeasure end
struct MeanKullbackLeiblerDivergence <: PathDistributionMeasure end

# Map structs to functions

# These return Rasters
graph_function(m::BetweennessKweighted) = betweenness_kweighted
graph_function(m::BetweennessQweighted) = betweenness_qweighted
graph_function(m::ConnectedHabitat) = connected_habitat
graph_function(m::Criticality) = criticality
# Returns a tuple
graph_function(m::EigMax) = eigmax
# These return scalars
graph_function(m::MeanLeastCostKullbackLeiblerDivergence) = mean_lc_kl_divergence
graph_function(m::MeanKullbackLeiblerDivergence) = mean_kl_divergence
# These return sparse arrays
graph_function(m::EdgeBetweennessKweighted) = edge_betweenness_kweighted
graph_function(m::EdgeBetweennessQweighted) = edge_betweenness_qweighted

# Map structs to function keywords, 
# a bit of a hack until we refactor the rest
keywords(gm::GraphMeasure, p::AbstractProblem) = 
    (; _keywords(gm)..., solver=solver(p))
keywords(gm::ConnectedHabitat, p::AbstractProblem) = 
    (; _keywords(gm)..., solver=solver(p), approx=connectivity_measure(p).approx) 

# A trait for connectivity requirement
struct NeedsConnectivity end
struct NoConnectivity end
needs_connectivity(::GraphMeasure) = NoConnectivity()
needs_connectivity(::BetweennessKweighted) = NeedsConnectivity()
needs_connectivity(::BetweennessKweighted) = NeedsConnectivity()
needs_connectivity(::EdgeBetweennessKweighted) = NeedsConnectivity()
needs_connectivity(::EigMax) = NeedsConnectivity()
needs_connectivity(::ConnectedHabitat) = NeedsConnectivity()

# compute
# This is where things actually happen
#
# Add dispatch on connectivity measure
compute(gm::GraphMeasure, p::AbstractProblem, g::Union{Grid,GridRSP}) = 
    compute(needs_connectivity(gm), gm, p, g)
function compute(::NeedsConnectivity,
    gm::GraphMeasure, 
    p::AbstractProblem, 
    g::Union{Grid,GridRSP}
)
    cm = p.connectivity_measure
    distance_transformation = cm.distance_transformation
    connectivity_function = ConScape.connectivity_function(cm)
    # Handle multiple distance transformations
    if distance_transformation isa NamedTuple
        map(distance_transformation) do dt
            graph_function(gm)(g; keywords(m, p)..., distance_transformation=dt, connectivity_function)
        end
    else
        graph_function(gm)(g; keywords(m, p)..., distance_transformation=dt, connectivity_function)
    end
end
function compute(::NoConnectivity,
    gm::GraphMeasure, 
    p::AbstractProblem, 
    g::Union{Grid,GridRSP}
) 
    graph_function(gm)(g; keywords(m, p)...)
end