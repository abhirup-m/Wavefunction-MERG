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
from tqdm.notebook import tqdm
from multiprocessing import Pool
from time import time
from operator import itemgetter


def getBasis(num_levels, nTot=-1):
    """ The argument num_levels is the total number of qubits
    participating in the Hilbert space. Function returns a basis
    of the classical states necessary to express any state as a 
    superposition. Members of the basis are lists such as 
    [0,0,0,0], [0,0,0,1],..., [1,1,01] and [1,1,1,1], where each
    character represents the configuration (empty or occupied) of
    each single-particle level
    """
    
    basis = []
    for char_arr in itertools.product(["0", "1"], repeat=num_levels):
        if nTot == -1 or sum([int(ch) for ch in char_arr]) == nTot:
            basis.append("".join(char_arr))
    
    return basis


def visualise_state(manyBodyBasis, stateDecomposition):
    """ Gives a handy visualisation for a many-body vector 'state'. manyBodyBasis is the complete
    basis for the associated system. For a state |up,dn> - |dn,up>, the visualisation is of the form
        up|dn       dn|up
        1           -1
    """
    state_string = "\t\t".join(["|".join([["0", "\u2191", "\u2193", "2"][int(basis_state[2 * i]) + 2 * int(basis_state[2 * i + 1])] 
                                          for i in range(len(basis_state) // 2)]) for basis_state in stateDecomposition.keys()])
    coeffs_string = "\t\t".join([str(np.round(coeff, 3)) for coeff in stateDecomposition.values()])
    return state_string+"\n"+coeffs_string


def getOperator(manyBodyBasis, int_kind, site_indices):
    """ Constructs a matrix operator given a prescription.
    manyBodyBasis is the set of all possible classical states.
    int_kind is a string that defines the qubit operators taking
    part in the operator. For eg.,'+-' means 'c^dag c'. 
    site_indices is a list that defines the indices of the states
    on whom the operators act. For rg., [0,1] means the operator
    is c^dag_0 c_1.
    """
    
    assert isinstance(manyBodyBasis, list)
    assert False not in [isinstance(item, str) for item in manyBodyBasis]
    # check that the number of qubit operators in int_kind matches the number of provided indices.
    assert isinstance(int_kind, str)
    assert False not in [k in ['+', '-', 'n', 'h'] for k in int_kind], "Interaction type not among +, - or n."
    
    # check that each operator in int_kind is from the set {+,-,n,h}, since these are the only ones we handle right now.
    assert isinstance(site_indices, list)
    assert False not in [isinstance(index, int) for index in site_indices]
    assert len(int_kind) == len(site_indices), "Number of site indices in term does not match number of provided interaction types."

    # initialises a zero matrix
    operator = np.zeros([len(manyBodyBasis), len(manyBodyBasis)])
    
    # Goes over all pairs of basis states |b1>, |b2> of the operator in order to obtain each matrix element <b2|O|b1>.
    for start_index, start_state in enumerate(manyBodyBasis):
        
        # get the action of 'int_kind' on the state b2
        end_state, mat_ele = applyTermOnBasisState(start_state, int_kind, site_indices)

        if end_state in manyBodyBasis:
            end_index = manyBodyBasis.index(end_state)
            operator[end_index][start_index] = mat_ele
    return operator
    

def get_computational_coefficients(basis, state):
    """ Given a general state and a complete basis, returns specifically those
    basis states that can express this general state as a superposition. Also returns
    the associated coefficients of the superposition.
    """
    assert len(basis) == len(state)
    decomposition = dict()
    for i,coeff in enumerate(state):
        if coeff != 0:
            decomposition[basis[i]] = coeff
    
    return decomposition


def innerProduct(state2, state1):
    """ Calculates the overlap <state2 | state1>.
    """
    innerProduct = sum([np.conjugate(state2[bstate]) * state1[bstate] for bstate in state1 if bstate in state2])
    return innerProduct
    
    
def matrixElement(finalState, operator, initState):
    """ Calculates the matrix element <final_state | operator | init_state> of an
    operator between the states initState and finalState  
    """
    intermediateState = applyOperatorOnState(initState, operator)
    matElement = innerProduct(finalState, intermediateState)
    return matElement 


def fermionicHamiltonian(manyBodyBasis, terms_list):
    """ Creates a matrix Hamiltonian from the specification provided in terms_list. terms_list is a dictionary
    of the form {['+','-']: [[1.1, [0,1]], [0.9, [1,2]], [2, [3,1]]], ['n']: [[1, [0]], [0.5, [1]], [1.2, [2]], [2, [3]]]}.
    Each key represents a specific type of interaction, such as c^dag c or n. The value associated with that key 
    is a nested list, of the form [g,[i_1,i_2,...]], where the inner list represents the indices of the particles 
    to whom those interactions will be applied, while the float value g in the outer list represents the strength 
    of that term in the Hamiltonian. For eg., the first key-value pair represents the interaction 
    1.1c^dag_0 c_1 + 0.9c^dag_1 c_2 + ..., while the second pair represents 1n_0 + 0.5n_1 + ...
    """
    
    # initialise a zero matrix
    hamlt = np.zeros([len(manyBodyBasis), len(manyBodyBasis)])

    # loop over all keys of the dictionary, equivalent to looping over various terms of the Hamiltonian
    for int_kind, val in terms_list.items():

        couplings = [t1 for t1,t2 in val]
        site_indices_all = [t2 for t1,t2 in val]

        # for each int_kind, pass the indices of sites to the get_operator function to create the operator 
        # for each such term
        hamlt += sum([coupling * getOperator(manyBodyBasis, int_kind, site_indices) for coupling, site_indices in tqdm(zip(couplings, site_indices_all), total=len(couplings), desc="Obtaining operators for " + int_kind + " .")])
    return np.array(hamlt)


def diagonalise(basis, hamlt):
    """ Diagonalise the provided Hamiltonian matrix.
    Returns all eigenvals and states.
    """
    
    E, v = scipy.linalg.eigh(hamlt)
    with Pool() as pool:
        workers = [pool.apply_async(get_computational_coefficients, (basis, v[:,i])) for i in range(len(E))]
        eigstates = [worker.get() for worker in tqdm(workers, desc="Expressing state in terms of basis.")]
    return E, eigstates


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
        elif (op == "+" and int(bstate[index]) == 1) or (op == "-" and int(bstate[index]) == 0):
            final_coeff *= 0
        else:
            final_coeff *= (-1) ** sum([int(ch) for ch in bstate[:index]])
            bstate = bstate[:index] + str(1 - int(bstate[index])) + (bstate[index+1:] if index + 1 < len(bstate) else '')

    return bstate, final_coeff


def applyOperatorOnState(initialState, terms_list, finalState=dict(), tqdmDesc=None):
    """ Applies a general operator on a general state. The general operator is specified through
    the terms_list parameter. The description of this parameter has been provided in the docstring
    of the get_fermionic_hamiltonian function.
    """

    # loop over all basis states for the given state, to see how the operator acts 
    # on each such basis state
    for bstate, coeff in tqdm(initialState.items(), disable=False, desc=tqdmDesc):

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

                if mod_coeff != 0:
                    try:
                        finalState[mod_bstate] += mod_coeff
                    except:
                        finalState[mod_bstate] = mod_coeff
                    print (bstate, mod_bstate, mod_coeff, coeff * coupling)
                    print ()
                           
    return finalState


def get_SIAM_hamiltonian(manyBodyBasis, num_bath_sites, couplings):
    """ Gives the string-based prescription to obtain a SIAM Hamiltonian:
    H = sum_k Ek n_ksigma + hop_strength sum_ksigma c^dag_ksigma c_dsigma + hc 
        + imp_Ed sum_sigma n_dsigma + imp_U n_dup n_ddn + imp_Bfield S_dz
    The coupling argument is a list that contains all the Hamiltonian parameters.
    Other parameters are self-explanatory. 
    """

    Ek, hop_strength, imp_U, imp_Ed, imp_Bfield, zerothsite_U = couplings

    # ensure the number of terms in the kinetic energy is equal to the number of bath sites provided
    assert len(Ek) == num_bath_sites

    # adjust dispersion to make room for spin degeneracy: (Ek1, Ek2) --> (Ek1,  Ek1,  Ek2,  Ek2)
    #                                                      k1   k2            k1up  k1dn  k2up  k2dn
    Ek = np.repeat(Ek, 2)

    # create kinetic energy term, by looping over all bath site indices 2,3,...,2*num_bath_sites+1,
    # where 0 and 1 are reserved for the impurity orbitals and must therefore be skipped.
    ham_KE = fermionicHamiltonian(manyBodyBasis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})

    # create the impurity-bath hopping terms, by looping over the up orbital indices i = 2, 4, 6, ..., 2*num_bath_sites,
    # and obtaining the corresponding down orbital index as i + 1. The four terms are c^dag_dup c_kup, h.c., c^dag_ddn c_kdn, h.c.
    ham_hop = (fermionicHamiltonian(manyBodyBasis, {'+-': [[hop_strength, [0, i]] for i in range(2, 2 * num_bath_sites + 2, 2)]}) 
               + fermionicHamiltonian(manyBodyBasis, {'+-': [[hop_strength, [i, 0]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
               + fermionicHamiltonian(manyBodyBasis, {'+-': [[hop_strength, [1, i + 1]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
               + fermionicHamiltonian(manyBodyBasis, {'+-': [[hop_strength, [i + 1, 1]] for i in range(2, 2 * num_bath_sites + 2, 2)]})
              )

    # create the impurity local terms for Ed, U and B
    ham_imp = (fermionicHamiltonian(manyBodyBasis, {'n': [[imp_Ed, [0]], [imp_Ed, [1]]]}) 
               + fermionicHamiltonian(manyBodyBasis, {'nn': [[imp_U, [0, 1]]]})
               + fermionicHamiltonian(manyBodyBasis, {'n': [[0.5 * imp_Bfield, [0]]]})
               + fermionicHamiltonian(manyBodyBasis, {'n': [[-0.5 * imp_Bfield, [1]]]})
              )

    ham_zerothsite = fermionicHamiltonian(manyBodyBasis, {'nn': [[zerothsite_U, [2, 3]]]})

    return ham_KE + ham_hop + ham_imp + ham_zerothsite


def getKondoHamiltonian(manyBodyBasis, num_bath_sites, couplings):
    """ Gives the string-based prescription to obtain a SIAM Hamiltonian:
    H = sum_k Ek n_ksigma + kondo J sum_12 vec S_d dot vec S_{12} 
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
    ham_KE = fermionicHamiltonian(manyBodyBasis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})

    # create the sum_k Sdz Skz term, by writing it in terms of number operators. 
    # The first line is n_dup sum_k Skz = n_dup sum_ksigma (-1)^sigma n_ksigma, sigma=(0,1).
    # The second line is -n_ddn sum_k Skz = -n_ddn sum_ksigma (-1)^sigma n_ksigma, sigma=(0,1).
    zz_terms = (sum([], [[kondo_J, [0, 2 * k1, 2 * k2]] for k1, k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)]) 
                + sum([], [[-kondo_J, [0, 2 * k1 + 1, 2 * k2 + 1]] for k1, k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)])
                + sum([], [[-kondo_J, [1, 2 * k1, 2 * k2]] for k1, k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)])
                + sum([], [[kondo_J, [1, 2 * k1 + 1, 2 * k2 + 1]] for k1, k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)])
               )
    Ham_zz = 0.25 * fermionicHamiltonian(manyBodyBasis, {'n+-': zz_terms})
    Ham_plus_minus = 0.5 * (fermionicHamiltonian(manyBodyBasis, {'+-+-': [[kondo_J, [0, 1, 2 * k1 + 1, 2 * k2]] for k1,k2 in itertools.product(range(1, num_bath_sites + 1), repeat=2)]}))

    H_Bfield = fermionicHamiltonian(manyBodyBasis, {'n': [[0.5 * B_field, [0]], [-0.5 * B_field, [1]]]})
    return ham_KE + Ham_zz + Ham_plus_minus + np.conj(np.transpose(Ham_plus_minus)) + H_Bfield


def getReducedDensityMatrix(genState, partiesRemain):
    """ Returns the reduced density matrix, given a state and a set of parties 
    partiesRemain that will not be traced over. The state is provided in the form
    of a set of coefficients and classical states. The calculation will happen through the expression
    rho = |psi><psi| =  sum_ij |psi^A_i>|psi^B_i><psi^A_j|<psi^B_j|,
    rho_A =  sum_ij  sum_{b_B} |psi^A_i><psi^A_j|<b_B|psi^B_i><psi^B_j|b_B>
    """

    if len(partiesRemain) == len(list(genState.keys())[0]):
        redDenMat = np.outer(list(genState.values()), list(genState.values()))
        return redDenMat
    
    get_substring = lambda string, sub_indices: "".join(itemgetter(*sub_indices)(string))
    
    remainBasis = getBasis(len(partiesRemain))
    
    # get the set of indices that will be traced over by taking the complement of the set partiesRemain.
    partiesTraced = [i for i in range(len(list(genState.keys())[0])) if i not in partiesRemain]      

    dictionary = {}
    for state, coeff in genState.items():
        lR = get_substring(state, partiesRemain)
        lT = get_substring(state, partiesTraced)
        if lT not in dictionary:
            dictionary[lT] = dict([(state, 0) for state in remainBasis])
        dictionary[lT][lR] += coeff
    
    redDenMat = np.zeros((len(remainBasis), len(remainBasis)))
    for lT in dictionary:
        redDenMat += np.outer(list(dictionary[lT].values()), list(dictionary[lT].values()))
            
    redDenMat /= np.trace(redDenMat)

    return redDenMat


def getEntanglementEntropy(genState, parties):
    """ Calculate entanglement entropy for the given state and the given parties.
    S(A) = trace(rho_A ln rho_A)
    """

    # get the reduced density matrix
    redDenMat = getReducedDensityMatrix(genState, parties)

    # get its spectrum
    eigvals,_ = scipy.linalg.eigh(redDenMat)

    # get its non-zero eigenvals
    nonzero_eigvals = eigvals[eigvals > 0]

    # calculate von Neumann entropy using the non-zero eigenvals
    entEntropy = -np.sum(nonzero_eigvals * np.log(nonzero_eigvals))

    return entEntropy


def getMutualInfo(genState, parties):
    """ Calculate mutual information between the given parties in the given state.
    I2(A:B) = S(A) + S(B) - S(AB). parties must be a two member array, where 
    the first(second) gives the indices for the members in A(B). One or both elements 
    can also be an array, if that party has multiple sites within it. For eg., 
    parties = [[0, 1], [2, 3]] would calculate the mutual information between the set 
    (0,1) and the set (2,3).
    """

    assert len(parties) == 2

    # get entanglement entropies for party A, party B and their union.
    S_A = getEntanglementEntropy(genState, parties[0])
    S_B = getEntanglementEntropy(genState, parties[1])
    S_AB = getEntanglementEntropy(genState, list(parties[0]) + list(parties[1]))

    return S_A + S_B - S_AB
