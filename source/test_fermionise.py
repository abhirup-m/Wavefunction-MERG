import pytest
import numpy as np
from fermionise import *
from itertools import product
import scipy.linalg

INT_KINDS_1LEVEL = ['+', '-', 'n', 'h']
INT_KINDS_2LEVEL = ["".join(l) for l in product(INT_KINDS_1LEVEL, repeat=2)]

@pytest.mark.parametrize("num_levels", range(1, 5))
def test_getBasis(num_levels):
    basis = getBasis(num_levels)
    assert len(basis) == 2**num_levels
    assert isinstance(basis, list)
    assert False not in [isinstance(state, str) for state in basis]


@pytest.mark.parametrize("int_kind", INT_KINDS_1LEVEL)
@pytest.mark.parametrize("site_indices", [[0]])
@pytest.mark.parametrize("num_levels", [1])
def test_getOperator_1level(int_kind, site_indices, num_levels):
    manybodyBasis = getBasis(num_levels)
    operator = getOperator(manybodyBasis, int_kind, site_indices)
    matchingMatrix = np.zeros([2, 2])
    matrixElementMaps = {'+': [1, 0], '-': [0, 1], 'n': [1, 1], 'h': [0, 0]}
    row, col = matrixElementMaps[int_kind]
    matchingMatrix[row][col] = 1
    assert isinstance(operator, np.ndarray)
    assert False not in np.equal(operator, matchingMatrix)


@pytest.mark.parametrize("int_kind", INT_KINDS_2LEVEL)
@pytest.mark.parametrize("site_indices", [[0, 1]])
@pytest.mark.parametrize("num_levels", [2])
def test_getOperator_2levels(int_kind, site_indices, num_levels):
    manybodyBasis = getBasis(num_levels)
    operator = getOperator(manybodyBasis, int_kind, site_indices)
    assert isinstance(operator, np.ndarray)
    matchingMatrix = np.zeros([4, 4])
    matrixElementMaps = {'nn': [3, 3], 'n+': [3, 2], 'n-': [2, 3], 'nh': [2, 2],
                         '+n': [3, 1], '++': [3, 0], '+-': [2, 1], '+h': [2, 0],
                         '-n': [1, 3], '-+': [1, 2], '--': [0, 3], '-h': [0, 2],
                         'hn': [1, 1], 'h+': [1, 0], 'h-': [0, 1], 'hh': [0,0]}
    row, col = matrixElementMaps[int_kind]
    matchingMatrix[row][col] = -1 if int_kind in ["-+", "--", "n-", "n+"] else 1
    assert False not in np.equal(operator, matchingMatrix)


@pytest.mark.parametrize("state", product((0, 0.5, 1), repeat=4))
@pytest.mark.parametrize("num_levels", [2])
def test_get_computational_coefficients(state, num_levels):
    manybodyBasis = getBasis(num_levels)
    decomposition = get_computational_coefficients(manybodyBasis, np.array(state))
    assert isinstance(decomposition, dict)
    for bstate, coeff in decomposition.items():
        assert coeff == state[manybodyBasis.index(bstate)]


# test against ground state of the Anderson molecule
@pytest.mark.parametrize("couplings, idn", [(([-2.5],  2,  6, -0.5,  0,  0), 1), 
                                           ((   [-1], 0,  0, -0.5,  0,  0), 2), 
                                           ((    [1], 0,  1,    0,  0,  0), 3), 
                                           (( [-0.5], 0,  0,    0, -1,  0), 4), 
                                           ((    [0], 0,  0,    1,  0, -1), 5), 
                                           ])
def test_Anderson_molecule(couplings, idn):
    Es, V, U, Ed, B, Ub = couplings
    manyBodyBasis = getBasis(4)
    hamlt = get_SIAM_hamiltonian(manyBodyBasis, 1, couplings)
    eigenvals, eigstates = diagonalise(hamlt)
    gsenergy = pytest.approx(eigenvals[0])
    decomposition = get_computational_coefficients(manyBodyBasis, eigstates[0])
    if idn == 1:
        matrix = np.array([[Ed + Es[0], np.sqrt(2) * V, np.sqrt(2) * V],
                           [np.sqrt(2) * V, 2 * Ed + U, 0],
                           [np.sqrt(2) * V, 0, 2 * Es[0]]])
        E, X = scipy.linalg.eigh(matrix)
        assert gsenergy == E[0]
        C_SS, C_20, C_02 = list(X[:,0])
        C_SS /= np.sqrt(2)
        assert decomposition["1001"] / decomposition["0110"] == pytest.approx(-1)
        assert decomposition["1001"] / decomposition["1100"] == pytest.approx(C_SS / C_20)
        assert decomposition["1001"] / decomposition["0011"] == pytest.approx(C_SS / C_02)
    if idn == 2:
        assert gsenergy == -3
        assert decomposition["1111"] == 1
    if idn == 3:
        assert gsenergy == 0
        assert decomposition["0000"] == 1
    if idn == 4:
        assert gsenergy == -1.5
        assert decomposition["1011"] == 1
    if idn == 5:
        assert gsenergy == -1
        assert decomposition["0011"] == 1


# test against ground state of the Anderson molecule (2 bath sites)
@pytest.mark.parametrize("couplings, idn", [(([-2.5, -2.5],  2,  6, -0.5,  0,  0), 1), 
                                           ((   [-1, -2.5], 0,  0, -0.5,  0,  0), 2), 
                                           ((    [1, -2.5], 0,  1,    0,  0,  0), 3), 
                                           (( [-0.5, -2.5], 0,  0,    0, -1,  0), 4), 
                                           ((    [0, -2.5], 0,  0,    1,  0, -1), 5), 
                                           ])


def test_Anderson_molecule_2sites(couplings, idn):
    Es, V, U, Ed, B, Ub = couplings
    manyBodyBasis = getBasis(6)
    hamlt = get_SIAM_hamiltonian(manyBodyBasis, 2, couplings)
    eigenvals, eigstates = diagonalise(hamlt)
    gsenergy = pytest.approx(eigenvals[0])
    decomposition = get_computational_coefficients(manyBodyBasis, eigstates[0])
    if idn == 1:
        matrix = np.array([[Ed + Es[0], 2 * V, 2 * V],
                           [2 * V, 2 * Ed + U, 0],
                           [2 * V, 0, 2 * Es[0]]])
        E, X = scipy.linalg.eigh(matrix)
        assert gsenergy == E[0] - 5
    if idn == 2:
        assert gsenergy == -3 - 5
        assert decomposition["111111"] == 1
    if idn == 3:
        assert gsenergy == 0 - 5
        assert decomposition["000011"] == 1
    if idn == 4:
        assert gsenergy == -1.5 - 5
        assert decomposition["101111"] == 1
    if idn == 5:
        assert gsenergy == -1 - 5
        assert decomposition["001111"] == 1


# test against ground state of the Kondo molecule
@pytest.mark.parametrize("couplings, idn", [(([-1], 5, 0), 1), 
                                            (([0.59, 1.2], 2, 0), 2),
                                           ])
def test_Kondo_molecule(couplings, idn):
    Es, J, B = couplings
    num_bath_sites = idn
    manyBodyBasis = getBasis(2 * (1 + num_bath_sites))
    hamlt = getKondoHamiltonian(manyBodyBasis, num_bath_sites, couplings)
    eigenvals, eigstates = diagonalise(hamlt)
    gsenergy = pytest.approx(eigenvals[0])
    decomposition = get_computational_coefficients(manyBodyBasis, eigstates[0])
    filling = getOperator(manyBodyBasis, 'n', [2])
    print (decomposition, get_operator_overlap(eigstates[0], eigstates[0], filling))
    if idn == 1:
        assert decomposition["1001"] / decomposition["0110"] == -1
        assert np.abs(decomposition["1001"]) == np.abs(decomposition["0110"]) == pytest.approx(1/np.sqrt(2))
        assert gsenergy == -3 * J / 4 + Es[0]
    if idn == 2:
        assert decomposition["100100"] / decomposition["011000"] == -1
        assert decomposition["100001"] / decomposition["010010"] == -1


@pytest.mark.parametrize("initialState, terms_list", [({"10": 1/np.sqrt(3), "01": -2/np.sqrt(3)}, 
                                                       {op: [[1, [0, 1]]]}) for op in ("-+", "hn", "+n")]
                         )
def test_applyOperatorOnState_2states(initialState, terms_list):
    finalState = applyOperatorOnState(initialState, terms_list, finalState=dict())
    if "-+" in terms_list:
        assert finalState == {"01": -1/np.sqrt(3)}
    if "hn" in terms_list:
        assert finalState == {"01": -2/np.sqrt(3)}
    if "+n" in terms_list:
        assert finalState == {"11": -2/np.sqrt(3)}


@pytest.mark.parametrize("initialState, terms_list", [({"110": 1/np.sqrt(3), "101": -2/np.sqrt(3)}, 
                                                       {'+-': [[1, [i, j]]]}) for i,j in [(0, 1), (1, 2), (2, 0)]]
                         )
def test_applyOperatorOnState_2states(initialState, terms_list):
    finalState = applyOperatorOnState(initialState, terms_list, finalState=dict())
    if (0, 1) == terms_list['+-'][0][1]:
        assert finalState == {}
    if (1, 2) == terms_list['+-'][0][1]:
        assert finalState == {"110": -2/np.sqrt(3)}
    if (2, 0) == terms_list['+-'][0][1]:
        assert finalState == {"011": -1/np.sqrt(3)}


@pytest.mark.parametrize("genState, parties", [({"10": 1/np.sqrt(5), "01": 2/np.sqrt(5)}, (0,) ),
                                               ({"101": 1/np.sqrt(4), "110": np.sqrt(2)/np.sqrt(4), "011": 1/np.sqrt(4)}, (0,) ),
                                               ({"101": 1/np.sqrt(4), "110": np.sqrt(2)/np.sqrt(4), "011": 1/np.sqrt(4)}, (0, 1) ),
                                               ]
                         )
def test_entanglemententropy(genState, parties):
    print (parties)
    entEntropy = pytest.approx(getEntanglementEntropy(genState, parties))
    if "10" in genState:
        assert entEntropy == -(0.2) * np.log(0.2) - (0.8) * np.log(0.8)
    if "101" in genState and parties == (0):
        assert entEntropy == -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
    if "101" in genState and parties == (0, 1):
        assert entEntropy == - 2 * 0.5 * np.log(0.5)
