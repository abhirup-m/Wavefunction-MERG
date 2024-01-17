using LinearAlgebra

function ReducedDensityMatrix(genState::Dict{String, Float64}, partiesRemain::Vector{Int64})

    if length(partiesRemain) == length(genState)
        redDenMat = values(genState) * collect(values(genState)')
        return redDenMat
    end
    getsubstring(s, indices) = join([ch for (i, ch) in enumerate(s) if i in indices])

    remainBasis = BasisStates(length(partiesRemain))

    partiesTraced = setdiff(1:length(collect(keys(genState))[1]), partiesRemain)

    stateTracedBasis = Dict()
    for (state, coeff) in genState
        labelRemain = getsubstring(state, partiesRemain)
        labelTraced = getsubstring(state, partiesTraced)
        if ! haskey(stateTracedBasis, labelTraced) 
            stateTracedBasis[labelTraced] = Dict([(state, 0.) for state in remainBasis])
    	end
        stateTracedBasis[labelTraced][labelRemain] += coeff
    end

    redDenMat = (+)([collect(values(stateTracedBasis[labelTraced])) * collect(values(stateTracedBasis[labelTraced]))'
		     for labelTraced in keys(stateTracedBasis)]...)

    return redDenMat
end

function EntanglementEntropy(genState, parties::Vector{Int64})
    redDenMat = ReducedDensityMatrix(genState, parties)

    eigenvalues = eigvals(Hermitian(redDenMat))

    nonzero_eigvals = eigenvalues[eigenvalues .> 0]

    entEntropy = -sum(nonzero_eigvals .* log.(nonzero_eigvals))

    partiesTraced = setdiff(1:length(collect(keys(genState))[1]), parties)
    redDenMat = ReducedDensityMatrix(genState, partiesTraced)

    eigenvalues_complement = eigvals(Hermitian(redDenMat))

    nonzero_eigvals_complement = eigenvalues_complement[eigenvalues_complement .> 0]

    entEntropy_complement = -sum(nonzero_eigvals_complement .* log.(nonzero_eigvals_complement))

    return 0.5 * (entEntropy + entEntropy_complement)
end

function MutualInfo(genState, partiesA::Vector{Int64}, partiesB::Vector{Int64})

    S_A = EntanglementEntropy(genState, partiesA)
    S_B = EntanglementEntropy(genState, partiesB)
    S_AB = EntanglementEntropy(genState, [partiesA; partiesB])

    return S_A + S_B - S_AB
end