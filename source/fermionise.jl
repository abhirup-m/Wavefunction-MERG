using IterTools, LinearAlgebra, ProgressBars

function BasisStates(numLevels, totOccupancy=-1)
	basisStates = [lpad(string(i, base=2), numLevels, '0') for i in 0:2^(numLevels)-1]
	basisStates = [state for state in basisStates if sum(parse.(Int, state)) == totOccupancy || totOccupancy == -1]
	return basisStates
end


function InnerProduct(genStateLeft, genStateRight)
	innerProduct = sum([conj(genStateLeft[key]) * genStateRight[key] for (key, value) in genStateRight])
    return innerProduct
end
    
    
function ComputationalCoefficients(basisStates, genState)
    @assert length(basisStates) == length(genState)
    stateDecompose = Dict([(basisStates[i], coeff) for (i, coeff) in enumerate(genState)])
    return stateDecompose
end


function MatrixElement(leftState, operatorList, rightState)
    intermediateState = ApplyOperatorOnState(rightState, operatorList)
    matrixElement = InnerProduct(leftState, intermediateState)
    return matrixElement 
end


function ApplyChunkOnBasisState(basisState, operatorChunk, siteIndices)

    @assert ! (false in [k in ['+', '-', 'n', 'h'] for k in operatorChunk]) "Interaction type not among +, - or n."
    @assert length(operatorChunk) == length(siteIndices) "Number of site indices in term does not match number of provided interaction types."

    prefactor = 1
    basisStateArray = [parse(Int, ch) for ch in split(basisState, "")]
    for (op, index) in zip(reverse(operatorChunk), reverse(siteIndices))
	siteOccupancy = basisStateArray[index]
        if op == 'n'
            prefactor *= siteOccupancy
    	elseif op == 'h'
            prefactor *= 1 - siteOccupancy
	elseif (op == '+' && siteOccupancy == 1) || (op == '-' && siteOccupancy == 0)
            prefactor *= 0
	    break
        else
	    prefactor *= (-1) ^ sum(basisStateArray[1:index-1])
	    basisStateArray[index] = 1 - siteOccupancy
        end
    end

    modifiedBasisState = join(basisStateArray)
    return modifiedBasisState, prefactor
end


function diagonalise(basisStates, matrix)
    eigenvals, eigenvecMatrix = eigen(matrix)
    eigenstateDecompositionArray = [ComputationalCoefficients(basisStates, eigenvector) for eigenvector in eachcol(eigenvecMatrix)]
    return eigenvals, eigenstateDecompositionArray
end


function ChunkOperatorMatrix(basisStates, operatorChunk, siteIndices)

    lengthBasis = length(basisStates)
    operatorMatrix = zeros((lengthBasis, lengthBasis))
    
    for (startIndex, startState) in enumerate(basisStates)
        endState, matrixElement = ApplyChunkOnBasisState(startState, operatorChunk, siteIndices)
	endIndex = findall(basisStates .== endState)[1]
	operatorMatrix[lengthBasis * (startIndex - 1) + endIndex] = matrixElement
    end
    return operatorMatrix
end


function ApplyOperatorOnState(genState, operatorList)
	finalState = Dict()
	for (basisState, basisCoefficient) in genState
		for (operatorChunk, chunkDescription) in operatorList
			for (couplingStrength, siteIndices) in chunkDescription
				modifiedBasisState, prefactor = ApplyChunkOnBasisState(basisState, operatorChunk, siteIndices)
				if haskey(finalState, modifiedBasisState)
					finalState[modifiedBasisState] += prefactor * basisCoefficient * couplingStrength
				else
					finalState[modifiedBasisState] = prefactor * basisCoefficient * couplingStrength
				end
			end
		end
	end
                           
	return finalState
end


function GeneralOperatorMatrix(basisStates, operatorList)
	lengthBasis = length(basisStates)
	generalOperatorMatrix = zeros((lengthBasis, lengthBasis))
	for (operatorChunk, chunkDescription) in operatorList
		generalOperatorMatrix += sum([coupling * ChunkOperatorMatrix(basisStates, operatorChunk, siteIndices) for (coupling, siteIndices) in tqdm(chunkDescription, total=length(chunkDescription))])
	end
	return generalOperatorMatrix
end


function KondoHamiltonian(basisStates, numBathSites, couplings)
	(kineticEnergy, kondoCoupling) = couplings
	@assert length(kineticEnergy) == numBathSites
	kineticEnergy = repeat(kineticEnergy, inner=2)
	upIndices = range(3, 2 * numBathSites + 1, step=2)
	downIndices = range(4, 2 * numBathSites + 2, step=2)

	kineticEnergyUp = Dict([("n", [[kineticEnergy[i - 2], [i]] for i in upIndices])])
	kineticEnergyDown = Dict([("n", [[kineticEnergy[i - 2], [i]] for i in downIndices])])
	kineticEnergyOperator = (GeneralOperatorMatrix(basisStates, kineticEnergyUp) +
				 GeneralOperatorMatrix(basisStates, kineticEnergyDown)
				 )

	zzUpUp = Dict([("n+-", [[kondoCoupling, [1, k1, k2]] for (k1, k2) in Iterators.product(upIndices, upIndices)])]) 
	zzUpDown = Dict([("n+-", [[-kondoCoupling, [1, k1, k2]] for (k1, k2) in Iterators.product(downIndices, downIndices)])]) 
	zzDownUp = Dict([("n+-", [[-kondoCoupling, [2, k1, k2]] for (k1, k2) in Iterators.product(upIndices, upIndices)])]) 
	zzDownDown = Dict([("n+-", [[kondoCoupling, [2, k1, k2]] for (k1, k2) in Iterators.product(downIndices, downIndices)])]) 
	zzOperator = 0.25 * (GeneralOperatorMatrix(basisStates, zzUpUp) +
			 GeneralOperatorMatrix(basisStates, zzUpDown) +
			 GeneralOperatorMatrix(basisStates, zzDownUp) +
			 GeneralOperatorMatrix(basisStates, zzDownDown)
			 )

	plusMinusOperator = 0.5 * (GeneralOperatorMatrix(basisStates, Dict([("+-+-", [[kondoCoupling, [1, 2, qdown, kup]] for (kup, qdown) in Iterators.product(upIndices, downIndices)])])))

	return kineticEnergyOperator  + zzOperator + plusMinusOperator + adjoint(plusMinusOperator)
end

function main(numBathSites)
	basisStates = BasisStates(2 * (1 + numBathSites))
	kineticEnergy = repeat([1], inner=numBathSites)
	fixedPointCouplings = [kineticEnergy, 10]
	KondoHam = KondoHamiltonian(basisStates, numBathSites, fixedPointCouplings)
	return
end
