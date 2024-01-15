function KondoHamiltonian(basisStates, numBathSites, couplings)
    if length(basisStates) == 0
        println("Chosen basis is empty, bailing out.")
        return
    end
	(kineticEnergy, kondoCoupling) = couplings
	@assert length(kineticEnergy) == numBathSites
	kineticEnergy = repeat(kineticEnergy, inner=2)
	upIndices = range(3, 2 * numBathSites + 1, step=2)

	operatorList = []
	
	for (k1, k2) in Iterators.product(upIndices, upIndices)
		push!(operatorList, (0.25 * kondoCoupling / numBathSites, "n+-", (1, k1, k2)))
		push!(operatorList, (-0.25 * kondoCoupling / numBathSites, "n+-", (1, k1 + 1, k2 + 1)))
		push!(operatorList, (-0.25 * kondoCoupling / numBathSites, "n+-", (2, k1, k2)))
		push!(operatorList, (0.25 * kondoCoupling / numBathSites, "n+-", (2, k1 + 1, k2 + 1)))
		push!(operatorList, (0.5 * kondoCoupling / numBathSites, "+-+-", (1, 2, k1 + 1, k2)))
		push!(operatorList, (0.5 * kondoCoupling / numBathSites, "+-+-", (2, 1, k1, k2 + 1)))
	end

    for upIndex in upIndices
		push!(operatorList, (kineticEnergy[upIndex - 2], "n", (upIndex,)))
		push!(operatorList, (kineticEnergy[upIndex - 1], "n", (upIndex + 1,)))
	end

	return GeneralOperatorMatrix(basisStates, operatorList)
end
