using LinearAlgebra

function ReducedDensityMatrix(genState::Dict{String, Float64}, partiesRemain::Vector{Int64})

    if length(partiesRemain) == length(genState)
        redDenMat = values(genState) * collect(values(genState)')
        return redDenMat
    end
    
    remainBasis = BasisStates(length(partiesRemain))
    
    partiesTraced = setdiff(1:length(collect(keys(genState))[1]), partiesRemain)

    stateTracedBasis = Dict()
    @showprogress for (state, coeff) in genState
        labelRemain = state[partiesRemain]
        labelTraced = state[partiesTraced]
        if ! haskey(stateTracedBasis, labelTraced) 
            stateTracedBasis[labelTraced] = Dict([(state, 0.) for state in remainBasis])
    	end
        stateTracedBasis[labelTraced][labelRemain] += coeff
    end
    
    redDenMat = (+)([collect(values(stateTracedBasis[labelTraced])) * collect(values(stateTracedBasis[labelTraced]))'
		     for labelTraced in keys(stateTracedBasis)]...)
            
    redDenMat /= tr(redDenMat)

    return redDenMat
end

function EntanglementEntropy(genState, parties::Vector{Int64})
    redDenMat = ReducedDensityMatrix(genState, parties)

    eigenvalues = eigvals(Hermitian(redDenMat))

    nonzero_eigvals = eigenvalues[eigenvalues .> 0]

    entEntropy = -sum(nonzero_eigvals .* log.(nonzero_eigvals))

    return entEntropy
end

function MutualInfo(genState, partiesA::Vector{Int64}, partiesB::Vector{Int64})

    S_A = EntanglementEntropy(genState, partiesA)
    S_B = EntanglementEntropy(genState, partiesB)
    S_AB = EntanglementEntropy(genState, [partiesA; partiesB])

    return S_A + S_B - S_AB
end