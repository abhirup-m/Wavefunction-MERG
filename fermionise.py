from itertools import product
import numpy as np
import scipy.linalg

def get_basis(num_levels):
    ## num_levels is total number of qubits participating in Hilbert space.
    ## many-body basis is constructed in the form of strings "0000", "0001",..., "1101, "1111",
    ## where each character represents the configuration (empty or occupied) of each single-particle level
    return [list(l) for l in product([0,1], repeat=num_levels)]

    
def get_operator(manybody_basis, int_kind, site_indices):
    assert False not in [k in ['+', '-', 'n', 'h'] for k in int_kind], "Interaction type not among +, - or n."
    assert len(int_kind) == len(site_indices), "Number of site indices in term does not match number of provided interaction types."
    operator = np.zeros([len(manybody_basis), len(manybody_basis)])
    for (i_1, b1), (i_2, b2) in product(enumerate(manybody_basis), repeat=2):
        modified_b2 = np.copy(b2)
        mat_ele = 1
        for op, index in zip(int_kind[::-1], site_indices[::-1]):
            if op == "n" or op == 'h':
                mat_ele *= b2[index] if op == "n" else 1 - b2[index]
            else:
                mat_ele *= (-1) ** sum(modified_b2[:index])
                if (op == "+" and modified_b2[index] == 1) or (op == "-" and modified_b2[index] == 0):
                    mat_ele = 0
                    break
                else:
                    modified_b2[index] = 1 - modified_b2[index]
        if False in np.equal(modified_b2, b1):
            mat_ele = 0
        operator[i_1][i_2] = mat_ele
    return operator
    
    
def get_fermionic_hamiltonian(manybody_basis, terms_list):
    ##     term_list is a dictionary of the form {['+','-']: [[1.1, [0,1]], [0.9, [1,2]], [2, [3,1]]], ['n']: [[1, [0]], [0.5, [1]], [1.2, [2]], [2, [3]]]}.
    ##     Each key represents a specific type of interaction, such as c^dag c or n. The value associated with that key
    ##     is a nested list, of the form [g,[i_1,i_2,...]], where the inner list represents the indices of the particles
    ##     to whom those interactions will be applied, while the float value g in the outer list represents the strength
    ##     of that term in the Hamiltonian. For eg., the first key-value pair represents the interaction 
    ##     1.1c^dag_0 c_1 + 0.9c^dag_1 c_2 + ..., while the second pair represents 1n_0 + 0.5n_1 + ...
    
    hamlt = np.zeros([len(manybody_basis), len(manybody_basis)])
    for int_kind, val in terms_list.items():
        couplings = [t1 for t1,t2 in val]
        site_indices_all = [t2 for t1,t2 in val]
        hamlt += sum([coupling * get_operator(manybody_basis, int_kind, site_indices) for coupling, site_indices in zip(couplings, site_indices_all)])
    return np.matrix(hamlt)


def diagonalise(hamlt):
    E, v = scipy.linalg.eigh(hamlt)
    return E, [v[:,i] for i in range(len(E))]


def get_operator_overlap(init_state, final_state, operator):
    return np.dot(np.transpose(final_state), np.dot(operator, init_state))


def get_spectral_function(basis, eigvals, eigstates, site_index, sigma, max_omega):
    omega_arr = np.linspace(-abs(max_omega), abs(max_omega), 1000)
    operator = get_operator(basis, '-', [site_index])
    ground_states = [eigs for i,eigs in enumerate(eigstates) if eigvals[i] == min(eigvals)]
    exc_states = [eigs for i,eigs in enumerate(eigstates) if eigvals[i] > min(eigvals)]
    deg = len(ground_states)
    spec_func = 0
    for (i_g, g_state), (i_e, ex_state) in product(enumerate(ground_states), enumerate(exc_states)):
        spec_func += np.abs(get_operator_overlap(ex_state, g_state, operator))**2 * np.exp(-(omega_arr + eigvals[i_g] - eigvals[i_e + deg])**2 / (2 * sigma**2))
        spec_func += np.abs(get_operator_overlap(g_state, ex_state, operator))**2 * np.exp(-(omega_arr - eigvals[i_g] + eigvals[i_e + deg])**2 / (2 * sigma**2))
    spec_func = spec_func / np.trapz(spec_func, omega_arr)
    return omega_arr, spec_func    
    
    
def get_computational_coefficients(basis, state):
    computational_basis = [np.concatenate((np.zeros(i), [1], np.zeros(len(basis) - 1 - i))) for i in range(len(basis))]
    coefficients = [np.round(np.inner(basis_state, state), 5) for basis_state in computational_basis]
    return [coeff for coeff, b in zip(coefficients, basis) if coeff != 0], [b for coeff, b in zip(coefficients, basis) if coeff != 0]


def visualise_state(mb_basis, state, state_labels=[]):
    computational_coeffs, basis_states = get_computational_coefficients(mb_basis, state)
    state_string = "\t".join(["|".join([["0", "\u2191", "\u2193", "2"][basis_state[2 * i] + 2 * basis_state[2 * i + 1]] for i in range(len(basis_state) // 2)]) for basis_state in basis_states])
    coeffs_string = "\t".join([str(np.round(coeff, 5)) for coeff in computational_coeffs])
    return state_string+"\n"+coeffs_string


def get_SIAM_hamiltonian(mb_basis, num_bath_sites, Ek, hop_strength, imp_U, imp_Ed, B_field=0):
    assert len(Ek) == num_bath_sites
    Ek = np.repeat(Ek, 2)
    ham_KE = get_fermionic_hamiltonian(mb_basis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})
    ham_hop = (get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [0, 2 * i]] for i in range(1, num_bath_sites + 1)]}) 
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [2 * i, 0]] for i in range(1, num_bath_sites + 1)]})
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [1, 2 * i + 1]] for i in range(1, num_bath_sites + 1)]})
               + get_fermionic_hamiltonian(mb_basis, {'+-': [[hop_strength, [2 * i + 1, 1]] for i in range(1, num_bath_sites + 1)]})
              )
    ham_imp = (get_fermionic_hamiltonian(mb_basis, {'n': [[imp_Ed, [0]], [imp_Ed, [1]]]}) 
               + get_fermionic_hamiltonian(mb_basis, {'nn': [[imp_U, [0, 1]]]})
               + get_fermionic_hamiltonian(mb_basis, {'n': [[B_field, [0]]]})
               + get_fermionic_hamiltonian(mb_basis, {'n': [[-B_field, [1]]]})
              )
    return ham_KE + ham_hop + ham_imp


def get_Kondo_hamiltonian(mb_basis, num_bath_sites, Ek, kondo_J, B_field=0):
    assert len(Ek) == num_bath_sites
    Ek = np.repeat(Ek, 2)
    ham_KE = get_fermionic_hamiltonian(mb_basis, {'n': [[Ek[i - 2], [i]] for i in range(2, 2 * num_bath_sites + 2)]})
    Ham_zz = 0.75 * (sum([get_fermionic_hamiltonian(mb_basis, {'nn': [[kondo_J * (-1)**i, [0, 2 + i]]]}) for i in range(2*num_bath_sites)]) 
                     + sum([get_fermionic_hamiltonian(mb_basis, {'nn': [[-kondo_J * (-1)**i, [1, 2 + i]]]}) for i in range(2*num_bath_sites)]))
    Ham_plus_minus = 0.5 * (get_fermionic_hamiltonian(mb_basis, {'+-+-': [[kondo_J, [0, 1, 2 * k1 + 1, 2 * k2]] for k1,k2 in product(range(1, num_bath_sites + 1), repeat=2)]}))
    H_Bfield = get_fermionic_hamiltonian(mb_basis, {'n': [[0.5 * B_field, [0]], [-0.5 * B_field, [1]]]})
    return ham_KE + Ham_zz + Ham_plus_minus + Ham_plus_minus.H + H_Bfield
