using Combinatorics
using Distributed
using LinearAlgebra
using SharedArrays
@everywhere using ProgressMeter
@everywhere using SharedArrays, Distributed

function BasisStates(numLevels::Int, totOccupancy=-999, totSpin=-999)
	basisStates = String[]
	for i in 0:2^(numLevels)-1
		state = lpad(string(i, base=2), numLevels, '0')
        stateTotOccupancy = sum([parse(Int, ch) for ch in state])
        stateTotSpin = 0.5 * (sum([parse(Int, ch) for ch in state[1:2:end]]) 
            - sum([parse(Int, ch) for ch in state[2:2:end]]))
		if ((totOccupancy == -999 || stateTotOccupancy == totOccupancy)
            && (totSpin == -999 || stateTotSpin == totSpin))
			push!(basisStates, state)
		end
	end
	return basisStates
end


function InnerProduct(genStateLeft::Dict{String, Float64}, genStateRight::Dict{String, Float64})
	innerProduct = sum([conj(genStateLeft[key]) * genStateRight[key] for (key, value) in genStateRight])
    return innerProduct
end
    
    
function ComputationalCoefficients(basisStates::Vector{String}, genState)
    @assert length(basisStates) == length(genState)
    stateDecompose = Dict([(basisStates[i], coeff) for (i, coeff) in enumerate(genState)])
    return stateDecompose
end


function MatrixElement(leftState::Dict{String, Float64}, operatorList, rightState::Dict{String, Float64})
    return InnerProduct(leftState, ApplyOperatorOnState(rightState, operatorList)) 
end


@everywhere function ApplyChunkOnBasisState(basisState::String, operatorChunk::String, siteIndices)
    @assert ! (false in [k in ['+', '-', 'n', 'h'] for k in operatorChunk]) "Interaction type not among +, - or n."
    @assert length(operatorChunk) == length(siteIndices) "Number of site indices in term does not match number of provided interaction types."

    prefactor::Float64 = 1
    basisStateArray = [parse(Int, ch) for ch in split(basisState, "")]
    for (op, index) in zip(reverse(operatorChunk), reverse(siteIndices))
    	siteOccupancy = basisStateArray[index]
        if op == 'n'
            prefactor *= siteOccupancy
    	elseif op == 'h'
            prefactor *= 1 - siteOccupancy
    	elseif (op == '+' && siteOccupancy == 1) || (op == '-' && siteOccupancy == 0)
            prefactor = 0
        else
    	    prefactor *= index == 1 ? 1 : (-1) ^ cumsum(basisStateArray)[index-1]
    	    basisStateArray[index] = 1 - siteOccupancy
        end
        if prefactor == 0
            break
        end
    end
    return join(basisStateArray), prefactor
end


function diagonalise(basisStates, matrix)
    if length(basisStates) == 0
        println("Chosen basis is empty, bailing out.")
        return
    end
    F = eigen(Hermitian(matrix), 1:2)
    eigenstateDecompositionArray = [ComputationalCoefficients(basisStates, eigenvector) for eigenvector in eachcol(F.vectors)]
    return F.values, eigenstateDecompositionArray
end


function ApplyOperatorOnState(genState, operatorList)
	finalState = Dict()
	for (basisState, basisCoefficient) in genState
		for (operatorChunk, couplingStrength, siteIndices) in eachrow(operatorList)
			modifiedBasisState, prefactor = ApplyChunkOnBasisState(basisState, operatorChunk, siteIndices)
			if haskey(finalState, modifiedBasisState)
				finalState[modifiedBasisState] += prefactor * basisCoefficient * couplingStrength
			else
				finalState[modifiedBasisState] = prefactor * basisCoefficient * couplingStrength
			end
		end
	end
                           
	return finalState
end

@everywhere function helper(args)
    (startIndex, basisStates, operatorList, lengthBasis) = args
    columnVector = zeros(lengthBasis)
    for operator in operatorList
        (couplingStrength, operatorChunk, siteIndices) = operator
        endState, prefactor = ApplyChunkOnBasisState(basisStates[startIndex], operatorChunk, siteIndices)
        if prefactor != 0
            endIndex = findall(basisStates .== endState)[1]
            columnVector[endIndex] += couplingStrength * prefactor
        end
    end
    return columnVector
end


function GeneralOperatorMatrix(basisStates, operatorList)
	lengthBasis = length(basisStates)
    argsList = [(startIndex, basisStates, operatorList, lengthBasis) for startIndex in 1:lengthBasis]
    generalOperatorMatrix = stack(@showprogress pmap(helper, argsList, batch_size=10))
	return generalOperatorMatrix
end
