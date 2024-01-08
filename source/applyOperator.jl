using JSON, LinearAlgebra, ProgressBars, Dates

function applyTermOnBasisState(bstate, int_kind, site_indices)
    site_indices = [x + 1 for x in site_indices]
    # check that the operator is composed only out of +,-,n,h
    @assert !(0 in [occursin(ch, "+-nh") for ch in int_kind])

    # check that the number of operators in int_kind matches the number of sites in site_indices.
    @assert length(int_kind) == length(site_indices)

    # final_coeff stores any factors that might emerge from applying the operator.
    final_coeff = 1

    # loop over all characters in the operator string, along with the corresponding site indices.
    for (op, index) in zip(reverse(int_kind), reverse(site_indices))
        # if the character is a number or a hole operator, just give the corresponding occupancy.
        if op == 'n'
            final_coeff *= parse(Int, bstate[index])
        elseif op == 'h'
                final_coeff *= 1 - parse(Int, bstate[index])
            # if the character is a create or annihilate operator, check if their is room for that.
            # If not, set final_coeff to zero. If there is, flip the occupancy of the site.
        elseif op == '+'
            if parse(Int, bstate[index]) == 1
                final_coeff *= 0
            else
                final_coeff *= (-1)^sum([parse(Int, ch) for ch in bstate[1:index-1]])
                bstate = string(bstate[1:index-1], "1", (if (index < length(bstate)) bstate[index+1:end] else "" end))
            end
        elseif op == '-'
            if parse(Int, bstate[index]) == 0
                final_coeff *= 0
            else
                final_coeff *= (-1)^sum([parse(Int, ch) for ch in bstate[1:index-1]])
                bstate = string(bstate[1:index-1], "0", (if (index < length(bstate)) bstate[index+1:end] else "" end))
            end
        end
    end
    return bstate, final_coeff

end


function applyOperatorOnState(initialState, terms_list, finalpath)
    
    finalState = copy(initialState)
    # loop over all basis states for the given state, to see how the operator acts 
    # on each such basis state
    
    t = Dates.datetime2unix(now())
    
    for (bstate, coeff) in initialState

        # loop over each term (for eg the list [[0.5,[0,1]], [0.4,[1,2]]]) in the full interaction,
        # so that we can apply each such chunk to each basis state.
        for (int_kind, val) in terms_list

            # loop over the various coupling strengths and index sets in each interaction term. In
            # the above example, coupling takes the values 0.5 and 0.4, while site_indices take the values
            # [0,1] and [1,2].
            for (coupling, site_indices) in val

                # apply each such operator chunk to each basis state
                mod_bstate, mod_coeff = applyTermOnBasisState(bstate, int_kind, site_indices)

                # multiply this result with the coupling strength and any coefficient associated 
                # with the initial state
                mod_coeff *= coeff * coupling
                
                if haskey(finalState, mod_bstate )
                    finalState[mod_bstate] += mod_coeff
                else                        
                    finalState[mod_bstate] = mod_coeff
                end
                if finalState[mod_bstate] == 0
                    delete!(finalState, mod_bstate)
                end
            end
        end
    end
    println(trunc(Int, Dates.datetime2unix(now()) - t))
    
    t = Dates.datetime2unix(now())
    totalNorm = norm(values(finalState))
    map!(x->x./totalNorm, values(finalState))
    JSON.print(open(finalpath, "w"), finalState)
    println(trunc(Int, Dates.datetime2unix(now()) - t))
end

# applyOperatorOnState(Dict([("10", 0.5), ("01", 0.5)]), Dict([("-+", [[1, [0, 1]]])]))
(arg1path, arg2path, finalpath) = ARGS
initialState = JSON.parsefile(arg1path)
terms_list = JSON.parsefile(arg2path)
applyOperatorOnState(initialState, terms_list, finalpath)
