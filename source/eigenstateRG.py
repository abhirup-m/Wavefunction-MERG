"""
OVERVIEW
This is a library for wavefunction renormalisation calculations with the URG.
The strategy adopted is as follows:
1. Perform the forward Hamiltonian RG to get the set of couplings along the RG flow.
2. Using the couplings at the IR fixed point, write down a zero-bandwidth Hamiltonian.
3. Diagonalise this Hamiltonian to get the IR ground state.
4. Express this ground state as a superposition of classical states. For eg:, 
   |up,dn> - |dn,up> = |10> - |01>, where {|10>,|01>} are the classical states and 
   {1,-1} are the coefficients.
5. Apply the inverse RG transformations on these classical states to generate a new 
   set of coefficients for each RG step going backwards.
6. These sets of coefficients constitute the wavefunction RG flow. Use these to 
   calculate various measures along the RG flow.

ASSUMPTIONS & CONVENTIONS
1. Fermionic impurity model Hamiltonian.
2. All operators must be composed of c^dag, c, n or 1-n, where each operator acts on a 
   single fermionic fock state.
3. The indexing of sites is as follows:
        d_up    d_dn    k1_up   k1_dn   k2_up   k2_dn   ...
        0       1       2       3       4       5       ...,
   where up,dn indicates spins and k1,k2 need not be momentum space but can be real
   space indices. That will be decided based on whether the provided Hamiltonian is
   in real or k-space.
"""


import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from tqdm import tqdm
from multiprocessing import Pool
from time import time
from operator import itemgetter
from source.fermionise import *


def init_wavefunction(hamlt, mb_basis):
    """ Generates the initial wavefunction at the fixed point by diagonalising
    the Hamiltonian provided as argument. Expresses the state as a superposition
    of various classical states, returns these states and the associated coefficients.
    No IOMS are taken into account at this point.
    """
   
    eigvals, eigstates = diagonalise(hamlt)
    print ("G-state energy:", eigvals[eigvals == min(eigvals)])
    print (visualise_state(mb_basis, eigstates[0]))
    
    # ensure that the ground state is not degenerate
    assert sum (eigvals == min(eigvals)) == 1, "Ground state is degenerate!"
    
    # get the classical states and the associated coefficients
    decomposition = get_computational_coefficients(mb_basis, eigstates[0] / np.linalg.norm(eigstates[0]))

    return decomposition
    

def applyInverseTransform(decomposition_old, num_entangled, etaFunc, alpha, IOMconfig, silent=True):
    """ Apply the inverse unitary transformation on a state. The transformation is defined
    as U = 1 + eta + eta^dag.
    """

    # expand the basis by inserting configurations for the IOMS, in preparation of applying 
    # the eta,eta^dag on them. 
    decomposition_old = dict([(key + str(IOMconfig) + str(IOMconfig), val) for key, val in decomposition_old.copy().items()])

    # obtain the appropriate eta and eta_dag for this step
    eta_dag, eta = etaFunc(alpha, num_entangled)

    decomposition_new = decomposition_old.copy()

    # get the action of eta and etadag by calling predefined functions
    if IOMconfig == 1:
        applyOperatorOnState(decomposition_old, eta, finalState=decomposition_new, silent=silent)
    else:
        applyOperatorOnState(decomposition_old, eta_dag, finalState=decomposition_new, silent=silent)

    total_norm = np.linalg.norm(list(decomposition_new.values()))
    decomposition_new = {k: v / total_norm for k, v in decomposition_new.items() if v != 0}


    return decomposition_new


def getWavefunctionRG(init_couplings, alpha_arr, num_entangled, num_IOMs, IOMconfigs, hamiltonianFunc, etaFunc, silent=True):
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

    assert len(IOMconfigs) >= num_IOMs, """Number of values of IOMconfigs is not enough for the
    requested number of reverse RG steps."""

    assert False not in [IOMconfig in (0, 1) for IOMconfig in IOMconfigs], """At least one of the 
    configs provided  for the IOMs is not 0 or 1. Config must be either 0 or 1, representing
    occupied or unoccupied."""

    # convert the string into function objects
    # hamiltonianFunc = eval(hamiltonianFunc)
    # etaFunc = eval(etaFunc)

    # get the basis of all classical states.
    mb_basis = getBasis(2 * (1 + num_entangled))
    
    # obtain the zero-bandwidth Hamiltonian at the IR
    hamlt = hamiltonianFunc(mb_basis, num_entangled, init_couplings)

    # obtain the superposition decomposition of the ground state
    decomposition_init = init_wavefunction(hamlt, mb_basis)
    
    # Initialise empty arrays to store the RG flow of the basis states and 
    # corresponding coefficients at each step of the reverse RG
    decomposition_arr = [decomposition_init]

    # loop over the values of alpha and apply the appropriate unitary for each value.
    for i, (alpha, IOMconfig) in tqdm(enumerate(zip(alpha_arr[:num_IOMs], IOMconfigs[:num_IOMs])), total=num_IOMs, desc="Applying inverse unitaries", disable=True):

        # obtain the renormalised coefficients and the new set of superposition states by passing the coefficients and states
        # of the previous step, the number of currently entangled states (num_entangled + i), the eta generating function and
        # the value of alpha at the present step
        decomposition_new = applyInverseTransform(decomposition_arr[-1], num_entangled + i, 
                                                                   etaFunc, alpha, IOMconfig, silent=silent)

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
        computations[computable] = [funcNameMaps[computable](decomposition, members) for decomposition in tqdm(decomposition_arr, total=len(decomposition_arr), desc="Computing {}".format(computable))]
    return computations


def getEtaKondo(alpha, num_entangled):
    """ Defines eta and eta dagger operators for the Kondo model.
    """

    eta_dag = {
            # The first interaction kind is Sdz c^dag qbeta c_kbeta. First two lines are for beta=up,
            # last two lines are for beta=down.
            "n+-": 
            [[alpha, [0, 2 * (num_entangled + 1), 2 * i]] for i in range(1, num_entangled + 1)] + 
            [[-alpha, [1, 2 * (num_entangled + 1), 2 * i]] for i in range(1, num_entangled + 1)] +
            [[-alpha, [0, 2 * (num_entangled + 1) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[alpha, [1, 2 * (num_entangled + 1) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)],
            # The second interaction kind is c^dag_dbetabar c_dbeta c^dag qbeta c_kbetabar. 
            # First line is for beta=up, last line is for beta=down.
            "+-+-": 
            [[alpha, [1, 0, 2 * (num_entangled + 1), 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[alpha, [0, 1, 2 * (num_entangled + 1) + 1, 2 * i]] for i in range(1, num_entangled + 1)]
            }
    # Simple the hermitian conjugate of each of the lines.
    eta = {"n+-": 
           [[alpha, [0, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[-alpha, [1, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] +
           [[-alpha, [0, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)] + 
           [[alpha, [1, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)],
           "+-+-": 
           [[alpha, [0, 1, 2 * i + 1, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[alpha, [1, 0, 2 * i, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)]
          }
    return eta_dag, eta
