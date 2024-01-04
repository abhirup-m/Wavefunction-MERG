######## TODO ########
#### incorporate other IOM sectors,
#### consider writing the data instead of storing it in memory,
#### parallelise,


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

def get_basis(num_levels):
    """ The argument num_levels is the total number of qubits
    participating in the Hilbert space. Function returns a basis
    of the classical states necessary to express any state as a 
    superposition. Members of the basis are lists such as 
    [0,0,0,0], [0,0,0,1],..., [1,1,01] and [1,1,1,1], where each
    character represents the configuration (empty or occupied) of
    each single-particle level
    """
    return ["".join(char_arr) for char_arr in itertools.product(["0", "1"], repeat=num_levels)]


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
    

def visualise_state(mb_basis, state):
    """ Gives a handy visualisation for a many-body vector 'state'. mb_basis is the complete
    basis for the associated system. For a state |up,dn> - |dn,up>, the visualisation is of the form
        up|dn       dn|up
        1           -1
    """

    decomposition = get_computational_coefficients(mb_basis, state)

    state_string = "\t\t".join(["|".join([["0", "\u2191", "\u2193", "2"][int(basis_state[2 * i]) + 2 * int(basis_state[2 * i + 1])] 
                                          for i in range(len(basis_state) // 2)]) for basis_state in decomposition.keys()])
    coeffs_string = "\t\t".join([str(np.round(coeff, 3)) for coeff in decomposition.values()])
    return state_string+"\n"+coeffs_string


def get_computational_coefficients(basis, state):
    """ Given a general state and a complete basis, returns specifically those
    basis states that can express this general state as a superposition. Also returns
    the associated coefficients of the superposition.
    """
    
    # For a general state of m particles, first generate the 2**m dimensional basis,
    # where the states are of the form [1,0,0,...0], [0,1,0,...,0], etc. 
    computational_basis = [np.concatenate((np.zeros(i), [1], np.zeros(len(basis) - 1 - i))) for i in range(len(basis))]
    
    # Obtain the contribution of each such basis state in the superposition by
    # calculating the overlap of each such state with the general state.
    coefficients = [np.round(np.inner(basis_state, state), 5) for basis_state in computational_basis]
    
    # filter out only those coefficients that are non-zero. 
    # also filter out the associated basis states.
    non_zero_coeffs = [coeff for coeff in coefficients if coeff != 0]
    non_zero_basis = [b for coeff, b in zip(coefficients, basis) if coeff != 0]

    decomposition = dict([(b, coeff) for coeff, b in zip(coefficients, basis) if coeff != 0])
    
    return decomposition


def get_operator(manybody_basis, int_kind, site_indices):

    """ Constructs a matrix operator given a prescription.
    manybody_basis is the set of all possible classical states.
    int_kind is a string that defines the qubit operators taking
    part in the operator. For eg.,'+-' means 'c^dag c'. 
    site_indices is a list that defines the indices of the states
    on whom the operators act. For rg., [0,1] means the operator
    is c^dag_0 c_1.
    """
    
    # check that the number of qubit operators in int_kind matches the number of provided indices.
    assert False not in [k in ['+', '-', 'n', 'h'] for k in int_kind], "Interaction type not among +, - or n."
    
    # check that each operator in int_kind is from the set {+,-,n,h}, since these are the only ones we handle right now.
    assert len(int_kind) == len(site_indices), "Number of site indices in term does not match number of provided interaction types."

    # initialises a zero matrix
    operator = np.zeros([len(manybody_basis), len(manybody_basis)])
    
    # Goes over all pairs of basis states |b1>, |b2> of the operator in order to obtain each matrix element <b2|O|b1>.
    for start_index, start_state in enumerate(manybody_basis):
        
        # get the action of 'int_kind' on the state b2
        end_state, mat_ele = applyTermOnBasisState(start_state, int_kind, site_indices)
                    
        end_index = manybody_basis.index(end_state)

        operator[end_index][start_index] = mat_ele
    return operator
    
    
def get_operator_overlap(init_state, final_state, operator):
    """ Calculates the overlap <final_state | operator | init_state>.
    """
    return np.dot(final_state.H, np.dot(operator, init_state))  
    
    
def get_fermionic_hamiltonian(manybody_basis, terms_list):
    """ Creates a matrix Hamiltonian from the specification provided in terms_list. terms_list is a dictionary
    of the form {['+','-']: [[1.1, [0,1]], [0.9, [1,2]], [2, [3,1]]], ['n']: [[1, [0]], [0.5, [1]], [1.2, [2]], [2, [3]]]}.
    Each key represents a specific type of interaction, such as c^dag c or n. The value associated with that key 
    is a nested list, of the form [g,[i_1,i_2,...]], where the inner list represents the indices of the particles 
    to whom those interactions will be applied, while the float value g in the outer list represents the strength 
    of that term in the Hamiltonian. For eg., the first key-value pair represents the interaction 
    1.1c^dag_0 c_1 + 0.9c^dag_1 c_2 + ..., while the second pair represents 1n_0 + 0.5n_1 + ...
    """
    
    # initialise a zero matrix
    hamlt = np.zeros([len(manybody_basis), len(manybody_basis)])

    # loop over all keys of the dictionary, equivalent to looping over various terms of the Hamiltonian
    for int_kind, val in terms_list.items():
        couplings = [t1 for t1,t2 in val]
        site_indices_all = [t2 for t1,t2 in val]

        # for each int_kind, pass the indices of sites to the get_operator function to create the operator 
        # for each such term
        hamlt += sum([coupling * get_operator(manybody_basis, int_kind, site_indices) for coupling, site_indices in zip(couplings, site_indices_all)])
    return np.matrix(hamlt)


def diagonalise(hamlt):
    """ Diagonalise the provided Hamiltonian matrix.
    Returns all eigenvals and states.
    """
    
    E, v = scipy.linalg.eigh(hamlt)
    # E, v = scipy.sparse.linalg.eigsh(hamlt, k=2, which='SA')
    return E, [v[:,i] for i in range(len(E))]


def applyTermOnBasisState(bstate, int_kind, site_indices):
    """ Applies a simple operator on a basis state. A simple operator is of the form '+-',[0,1].
    The first string, passed through the argument int_kind, indicates the form of the operator.
    It can be any operator of any length composed of the characters +,-,n,h. The list [0,1], passed
    through the argument site_indices, defines the indices of the sites on which the operators will 
    be applied. The n^th character of the string will act on the n^th element of site_indices. The
    operator is simple in the sense that there is no summation of multiple operators involved here.
    """

    # check that the operator is composed only out of +,-,n,h
    assert False not in [k in ['+', '-', 'n', 'h'] for k in int_kind], "Interaction type not among +, - or n."

    # check that the number of operators in int_kind matches the number of sites in site_indices.
    assert len(int_kind) == len(site_indices), "Number of site indices in term does not match number of provided interaction types."

    # final_coeff stores any factors that might emerge from applying the operator.
    final_coeff = 1

    # loop over all characters in the operator string, along with the corresponding site indices.
    for op, index in zip(int_kind[::-1], site_indices[::-1]):

        # if the character is a number or a hole operator, just give the corresponding occupancy.
        if op == "n":
            final_coeff *= int(bstate[index])
        elif op == "h":
            final_coeff *= 1 - int(bstate[index])

        # if the character is a create or annihilate operator, check if their is room for that.
        # If not, set final_coeff to zero. If there is, flip the occupancy of the site.
        elif op == "+":
            if int(bstate[index]) == 1:
                final_coeff *= 0
            else:
                bstate = bstate[:index] + '1' + (bstate[index+1:] if index + 1 < len(bstate) else '')
        elif op == "-":
            #print (bstate, site_indices)
            if int(bstate[index]) == 0:
                final_coeff *= 0
            else:
                bstate = bstate[:index] + '0' + (bstate[index+1:] if index + 1 < len(bstate) else '')

    return bstate, final_coeff
        

def applyOperatorOnState(decomposition_old, decomposition_new, terms_list):
    """ Applies a general operator on a general state. The general operator is specified through
    the terms_list parameter. The description of this parameter has been provided in the docstring
    of the get_fermionic_hamiltonian function.
    """

    # loop over all basis states for the given state, to see how the operator acts 
    # on each such basis state
    for bstate, coeff in tqdm(decomposition_old.items(), disable=True):

        # loop over each term (for eg the list [[0.5,[0,1]], [0.4,[1,2]]]) in the full interaction,
        # so that we can apply each such chunk to each basis state.
        for int_kind, val in terms_list.items():

            # loop over the various coupling strengths and index sets in each interaction term. In
            # the above example, coupling takes the values 0.5 and 0.4, while site_indices take the values
            # [0,1] and [1,2].
            for coupling, site_indices in val:

                # apply each such operator chunk to each basis state
                mod_bstate, mod_coeff = applyTermOnBasisState(bstate, int_kind, site_indices)

                # multiply this result with the coupling strength and any coefficient associated 
                # with the initial state
                mod_coeff *= coeff * coupling

                try:
                    decomposition_new[mod_bstate] += mod_coeff
                except:
                    decomposition_new[mod_bstate] = mod_coeff
                if decomposition_new[mod_bstate] == 0:
                    del decomposition_new[mod_bstate]

    return decomposition_new


def applyInverseTransform(decomposition_old, num_entangled, etaFunc, alpha, IOMconfig):
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
        decomposition_new = applyOperatorOnState(decomposition_old, decomposition_new, eta)
    else:
        decomposition_new = applyOperatorOnState(decomposition_old, decomposition_new, eta_dag)

    total_norm = np.linalg.norm(list(decomposition_new.values()))


    return decomposition_new


def getWavefunctionRG(init_couplings, alpha_arr, num_entangled, num_IOMs, IOMconfigs, hamiltonianFunc, etaFunc):
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
    hamiltonianFunc = eval(hamiltonianFunc)
    etaFunc = eval(etaFunc)

    # get the basis of all classical states.
    mb_basis = get_basis(2 * (1 + num_entangled))
    
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
                                                                   etaFunc, alpha, IOMconfig)

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
        # iterator = tqdm(zip(coefficients_arr, combinations_arr, itertools.repeat(members)), total=len(coefficients_arr), desc="Computing {}".format(computable))
        # computations[computable] = Pool().starmap(funcNameMaps[computable], iterator)
        # iterator = zip(coefficients_arr, combinations_arr, itertools.repeat(members))#, total=len(coefficients_arr), desc="Computing {}".format(computable))
        computations[computable] = [funcNameMaps[computable](list(decomposition.values()), list(decomposition.keys()), members) for decomposition in tqdm(decomposition_arr, total=len(decomposition_arr), desc="Computing {}".format(computable))]
    return computations


def get_SIAM_hamiltonian(mb_basis, num_bath_sites, couplings):
    """ Gives the string-based prescription to obtain a SIAM Hamiltonian:
    H = sum_k Ek n_ksigma + hop_strength sum_ksigma c^dag_ksigma c_dsigma + hc 
        + imp_Ed sum_sigma n_dsigma + imp_U n_dup n_ddn + imp_Bfield S_dz
    The coupling argument is a list that contains all the Hamiltonian parameters.
    Other parameters are self-explanatory. 
    """

    Ek, hop_strength, imp_U, imp_Ed, imp_Bfield = couplings

    # ensure the number of terms in the kinetic energy is equal to the number of bath sites provided
    assert len(Ek) == num_bath_sites

    # adjust dispersion to make room for spin degeneracy: (Ek1, Ek2) --> (Ek1,  Ek1,  Ek2,  Ek2)
    #                                                      k1   k2            k1up  k1dn  k2up  k2dn
    Ek = np.repeat(Ek, 2)

    # create kinetic energy term, by looping over all bath site indices 2,3,...,2*num_bath_sites+1,
    # where 0 and 1 are reserved for the impurity orbitals and must therefore be skipped.
    ham_KE = get_fermionic_hamiltonian(mb_basis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})

    # create the impurity-bath hopping terms, by looping over the up orbital indices i = 2, 4, 6, ..., 2*num_bath_sites,
    # and obtaining the corresponding down orbital index as i + 1. The four terms are c^dag_dup c_kup, h.c., c^dag_ddn c_kdn, h.c.
    ham_hop = (get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [0, i]] for i in range(2, 2 * num_bath_sites + 2, 2)]}) 
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [i, 0]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [1, i + 1]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [i + 1, 1]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
              )

    # create the impurity local terms for Ed, U and B
    ham_imp = (get_fermionic_hamiltonian(mb_basis, {'n': [[imp_Ed, [0]], [imp_Ed, [1]]]}) 
               + get_fermionic_hamiltonian(mb_basis, {'nn': [[imp_U, [0, 1]]]})
               + get_fermionic_hamiltonian(mb_basis, {'n': [[imp_Bfield, [0]]]})
               + get_fermionic_hamiltonian(mb_basis, {'n': [[-imp_Bfield, [1]]]})
              )
    return ham_KE + ham_hop + ham_imp


def get_Kondo_hamiltonian(mb_basis, num_bath_sites, couplings):
    """ Gives the string-based prescription to obtain a SIAM Hamiltonian:
    H = sum_k Ek n_ksigma + kondo J \sum_12 vec S_d dot vec S_{12} 
        + imp_Bfield S_dz
    The coupling argument is a list that contains all the Hamiltonian parameters.
    Other parameters are self-explanatory. 
    """

    Ek, kondo_J, B_field = couplings

    # ensure the number of terms in the kinetic energy is equal to the number of bath sites provided
    assert len(Ek) == num_bath_sites

    # adjust dispersion to make room for spin degeneracy: (Ek1, Ek2) --> (Ek1,  Ek1,  Ek2,  Ek2)
    #                                                      k1   k2            k1up  k1dn  k2up  k2dn
    Ek = np.repeat(Ek, 2)

    # create kinetic energy term, by looping over all bath site indices 2,3,...,2*num_bath_sites+1,
    # where 0 and 1 are reserved for the impurity orbitals and must therefore be skipped.
    ham_KE = get_fermionic_hamiltonian(mb_basis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})

    # create the sum_k Sdz Skz term, by writing it in terms of number operators. 
    # The first line is n_dup sum_k Skz = n_dup\sum_ksigma (-1)^sigma n_ksigma, sigma=(0,1).
    # The second line is -n_ddn sum_k Skz = -n_ddn\sum_ksigma (-1)^sigma n_ksigma, sigma=(0,1).
    Ham_zz = 0.25 * (sum([get_fermionic_hamiltonian(mb_basis, {'nn': [[kondo_J * (-1)**i, [0, 2 + i]]]}) for i in range(2*num_bath_sites)]) 
                     + sum([get_fermionic_hamiltonian(mb_basis, {'nn': [[-kondo_J * (-1)**i, [1, 2 + i]]]}) for i in range(2*num_bath_sites)]))

    # create the sum_12 Sd^+ S_12^- term, by writing it in terms of c-operators:
    # Sd^+ S_12^- = c^dag_dup c_ddn c^dag_k1dn c_k2up
    Ham_plus_minus = 0.5 * (get_fermionic_hamiltonian(mb_basis, {'+-+-': [[kondo_J, [0, 1, 2 * k1 + 1, 2 * k2]] for k1,k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)]}))

    H_Bfield = get_fermionic_hamiltonian(mb_basis, {'n': [[0.5 * B_field, [0]], [-0.5 * B_field, [1]]]})
    return ham_KE + Ham_zz + Ham_plus_minus + Ham_plus_minus.H + H_Bfield


def getReducedDensityMatrix(coeffs, bstates, partiesRemain):
    """ Returns the reduced density matrix, given a state and a set of parties 
    partiesRemain that will not be traced over. The state is provided in the form
    of a set of coefficients and classical states. The calculation will happen through the expression
    rho = |psi><psi| = \sum_ij |psi^A_i>|psi^B_i><psi^A_j|<psi^B_j|,
    rho_A = \sum_ij \sum_{b_B} |psi^A_i><psi^A_j|<b_B|psi^B_i><psi^B_j|b_B>
    """

    # get the set of indices that will be traced over by taking the complement of the set partiesRemain.
    partiesTraced = [i for i in range(len(bstates[0])) if i not in partiesRemain]

    # initialise a zero matrix for the reduced density matrix
    redDenMat = np.zeros([2**len(partiesRemain), 2**len(partiesRemain)])

    # get a classical basis for the reduced Hilbert space of the remaining parties.
    red_basis = get_basis(len(partiesRemain))

    # loop over all paris of basis states of the reduced space, and calcualte 
    # matrix element for each such pair in order to construct the full reduced dmatrix.
    for (i1, red_bstate1), (i2, red_bstate2) in itertools.product(enumerate(red_basis), repeat=2):

        # for each pair of basis states, loop over basis states of the full Hilbert space.
        for (j1, bstate1), (j2, bstate2) in itertools.product(enumerate(bstates), repeat=2):

            # This is explained by the formula for rho_A provided above.
            if ("".join(itemgetter(*partiesRemain)(bstate1)) == red_bstate1 and
                "".join(itemgetter(*partiesRemain)(bstate2)) == red_bstate2
                ):
                if (len(partiesTraced ) > 0 and "".join(itemgetter(*partiesTraced)(bstate1)) == "".join(itemgetter(*partiesTraced)(bstate2))
                ) or len(partiesTraced ) == 0:
                    redDenMat[i1][i2] += np.conjugate(coeffs[j1]) * coeffs[j2]

    redDenMat /= np.trace(redDenMat)
    return redDenMat


def getEntanglementEntropy(coeffs, bstates, parties):
    """ Calculate entanglement entropy for the given state and the given parties.
    S(A) = trace(rho_A ln rho_A)
    """

    # get the reduced density matrix
    redDenMat = getReducedDensityMatrix(coeffs, bstates, parties)
    # print (redDenMat)

    # get its spectrum
    eigvals,_ = scipy.linalg.eigh(redDenMat)

    # get its non-zero eigenvals
    nonzero_eigvals = eigvals[eigvals > 0]

    # calculate von Neumann entropy using the non-zero eigenvals
    entEntropy = -np.sum(nonzero_eigvals * np.log(nonzero_eigvals))

    #print (entEntropy)
    return entEntropy


def getMutualInfo(coeffs, bstates, parties):
    """ Calculate mutual information between the given parties in the given state.
    I2(A:B) = S(A) + S(B) - S(AB). parties must be a two member array, where 
    the first(second) gives the indices for the members in A(B). One or both elements 
    can also be an array, if that party has multiple sites within it. For eg., 
    parties = [[0, 1], [2, 3]] would calculate the mutual information between the set 
    (0,1) and the set (2,3).
    """

    assert len(parties) == 2

    # get entanglement entropies for party A, party B and their union.
    S_A = getEntanglementEntropy(coeffs, bstates, parties[0])
    S_B = getEntanglementEntropy(coeffs, bstates, parties[1])
    S_AB = getEntanglementEntropy(coeffs, bstates, list(parties[0]) + list(parties[1]))

    return S_A + S_B - S_AB


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


