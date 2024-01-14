def init_wavefunction(hamlt, mb_basis, displayGstate=False):
    """ Generates the initial wavefunction at the fixed point by diagonalising
    the Hamiltonian provided as argument. Expresses the state as a superposition
    of various classical states, returns these states and the associated coefficients.
    No IOMS are taken into account at this point.
    """
   
    eigvals, eigstates = diagonalise(mb_basis, hamlt)
    tolerance = 10
    print (eigvals[:2])
    print ("G-state energy:", eigvals[eigvals == min(eigvals)])
    if sum (np.round(eigvals, tolerance) == min(np.round(eigvals, tolerance))) == 1:
        gstate = eigstates[0]
    else:
        assert False, "Ground state is degenerate! No SU(2)-symmetric ground state exists."
    
    if displayGstate:
        print (visualise_state(mb_basis, gstate))

    return gstate
    

def applyInverseTransform(decomposition_old, num_entangled, etaFunc, alpha):
    """ Apply the inverse unitary transformation on a state. The transformation is defined
    as U = 1 + eta + eta^dag.
    """

    # expand the basis by inserting configurations for the IOMS, in preparation of applying 
    # the eta,eta^dag on them. 
    decomposition_old = dict([(key + "1100", val) for key, val in decomposition_old.copy().items()])

    # obtain the appropriate eta and eta_dag for this step
    eta_dag, eta = etaFunc(alpha, num_entangled)

    decomposition_new_eta = decomposition_old.copy()
    decomposition_new_etadag = decomposition_old.copy()
    
    with Pool(1) as pool:
        worker_eta = pool.apply_async(applyOperatorOnState, (decomposition_old, eta),
                              kwds={'finalState': decomposition_new_eta, 
                                    'tqdmDesc': "Applying eta, size=" + str(num_entangled)})
        worker_etadag = pool.apply_async(applyOperatorOnState, (decomposition_old, eta_dag),
                              kwds={'finalState': decomposition_new_etadag,
                                    'tqdmDesc': "Applying eta^dag, size=" + str(num_entangled)})
        decomposition_new_eta = worker_eta.get()
        decomposition_new_etadag = worker_etadag.get()
        
        
    decomposition_new_total = decomposition_new_eta.copy()
    decomposition_new_total.update(decomposition_new_etadag)
    
    total_norm = np.linalg.norm(list(decomposition_new_total.values()))

    decomposition_new_total = {k: v / total_norm for k, v in decomposition_new_total.items() if np.abs(v) / total_norm > 1e-5}


    return decomposition_new_total


def getWavefunctionRG(init_couplings, alpha_arr, num_entangled, num_IOMs, hamiltonianFunc, etaFunc, displayGstate=False):
    """ Manager function for obtaining wavefunction RG. 
    1. init_couplings is the set of couplings that are sufficient to construct the IR Hamiltonian
       and hence the ground state.
    2. alpha_arr is the array of Greens function denominators that will be used to construct the 
       eta operators at various steps of the RG. Each element of alpha_arr can itself be an array. 
    3. num_entangled is the number of states in the bath that are intended to be part of the emergent 
       window at the IR.
    4. num_IOMs is the number of states in the bath that we wish to re-entangle eventually.
    5. hamiltonianFunc is a string containing the name of a function with a definition 
       func(mb_basis, num_entangled, init_couplings) that creates a Hamiltonian matrix for the model 
       at the IR. It must be defined on a per-model basis.
    6. etaFunc is a string containing the name of a function with a definition etaFunc(alpha, num_entangled)
       that returns the eta and eta^dag operators given the step-dependent parameters alpha and num_entangled.
       This function must be defined on a per-model basis.
    """
    
    # make sure there are sufficient number of values provided in alpha_arr to re-entangled num_IOMs.
    assert len(alpha_arr) >= num_IOMs, """Number of values of 'alpha' is not enough for the
    requested number of reverse RG steps."""
    
    # convert the string into function objects
    # hamiltonianFunc = eval(hamiltonianFunc)
    # etaFunc = eval(etaFunc)

    # get the basis of all classical states.
    mb_basis = getBasis(2 * (1 + num_entangled))
    
    # obtain the zero-bandwidth Hamiltonian at the IR
    hamlt = hamiltonianFunc(mb_basis, num_entangled, init_couplings)

    # obtain the superposition decomposition of the ground state
    decomposition_init = init_wavefunction(hamlt, mb_basis, displayGstate=displayGstate)
    
    # Initialise empty arrays to store the RG flow of the basis states and 
    # corresponding coefficients at each step of the reverse RG
    decomposition_arr = [decomposition_init]
    
    # loop over the values of alpha and apply the appropriate unitary for each value.
    for i, alpha in tqdm(enumerate(alpha_arr[:num_IOMs]), total=num_IOMs, desc="Applying inverse unitaries", disable=True):

        # obtain the renormalised coefficients and the new set of superposition states by passing the coefficients and states
        # of the previous step, the number of currently entangled states (num_entangled + i), the eta generating function and
        # the value of alpha at the present step
        decomposition_new = applyInverseTransform(decomposition_arr[-1], num_entangled + 2 * i,
                                                  etaFunc, alpha)

        # append new results to full array
        decomposition_arr.append(decomposition_new)

    return decomposition_arr


def computations(decomposition_arr, computables):
    """ Perform various computations by passing the wavefunction RG data.
    The computables argument is a dictionary, of the examplary form
    {"VNE": [0,1], "I2": [[0,1],[2,3]]}. Each key is a quantity to calculate,
    and the list in the value of the key is the set of indices with whom to
    perform the calculation. For eg., the first key asks to calculate the 
    von Neumann entanglement entropy for the set of indices (0,1).
    """

    # initialise a dictionary with the already-provided keys to store the results
    computations = dict.fromkeys(computables.keys())

    # dictionary for mapping computable names to corresponding functions,
    funcNameMaps = {"VNE": getEntanglementEntropy,
                     "I2": getMutualInfo,
                     }

    # loop over all the computables that have been required
    # for each computable, loop over the coefficient RG flow
    # and perform the computation at every RG step.
    for computable, members in computables.items():
        computations[computable] = [funcNameMaps[computable](decomposition, members) for decomposition in tqdm(decomposition_arr, total=len(decomposition_arr), disable=False, desc="Computing {}".format(computable))]
    return computations


def getEtaKondo(alpha, num_entangled):
    """ Defines eta and eta dagger operators for the Kondo model.
    """

    eta_dag = {
            # The first interaction kind is Sdz c^dag qbeta c_kbeta. First two lines are for beta=up,
            # last two lines are for beta=down.
            "n+-":
            [[0.25 * alpha, [0, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] + 
            [[-0.25 * alpha, [1, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] +
            [[-0.25 * alpha, [0, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.25 * alpha, [1, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)],
            # The second interaction kind is c^dag_dbetabar c_dbeta c^dag qbeta c_kbetabar. 
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[0.5 * alpha, [1, 0, 2 * (num_entangled + 2), 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.5 * alpha, [0, 1, 2 * (num_entangled + 2) + 1, 2 * i]] for i in range(1, num_entangled + 1)]
            }
    # Simply the hermitian conjugate of each of the lines.
    eta = {"n+-": 
           [[0.25 * alpha, [0, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[-0.25 * alpha, [1, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] +
           [[-0.25 * alpha, [0, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)] + 
           [[0.25 * alpha, [1, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)],
           "+-+-": 
           [[0.5 * alpha, [0, 1, 2 * i + 1, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[0.5 * alpha, [1, 0, 2 * i, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)]
          }
    return eta_dag, eta
