#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import copy
import shutil
try:
    import fcntl
except ImportError:
    print "NO FILE LOCKING AVAILABLE (no fcntl on windows...)"
import scipy as sp
import scipy.linalg as la
import sympy as sy
import evoMPS.tdvp_uniform as tdvp
import evoMPS.dynamics as dy

try:
    import matplotlib.cm as cm
except ImportError:
    print "Matplotlib error..."

RES_LOAD_FROM = 'petal_res.txt'
use_CUDA = False

cols = {'G':        0,
        'g2inv':    1, 
        'D':        2, 
        'L':        3,
        'max_ir':   4,
        'eta':      5,
        'max_ir_oc':6,
        'energy':   7, 
        'cl':       8, 
        'entr_max': 9, 
        'U_av':     10, 
        'U_0':      11,
        'U2_0':     12,
        'P_av':     13,
        'P_0':      14,
        'P2_0':     15,
        'ReUUH_av': 16,
        'ReUUH_0':  17,
        'ReUUH2_0': 18,
        'wv_dom':   19, 
        'wv_fit':   20, 
        'gap':      21, 
        'exc_fn':   22,
        'fn':       23}
        
PARAM = 'g2inv'
PARAM_TEX = 'g^{-2}'
        
num_floatcols = len(cols) - 2

def get_ops(G, max_irrep):
    if G == 1:
        return U1_get_ops(max_irrep)
    if G == 2:
        return SU2_get_ops(max_irrep)

def U1_get_ops(max_n):
    d = 2 * max_n + 1
    U = sp.diag(sp.repeat(1. + 0.j, d - 1), k=-1)
    P = sp.diag(sp.arange(-max_n, +max_n + 1) * (1. + 0.j), k=0)
    P2 = sp.diag(sp.diag(P)**2)
    
    ReUUH = 0.5 * (sp.kron(U, U.conj().T) 
                   + sp.kron(U.conj().T, U)).reshape(d, d, d, d)
                   
    ReUUH_tp = [[0.5 * U, U.conj().T],
                [0.5 * U.conj().T, U]]
                   
    U = [[U]]
    P = [P]
                   
    return U, P, P2, ReUUH, ReUUH_tp
    
def U1_get_L(theta, max_n=5):
    L = sp.diag(sp.exp(-1.j * theta * sp.arange(-max_n, +max_n + 1)), k=0)
    return L
    
def U1_get_theta(max_n=5, pos_theta=False):
    d = 2 * max_n + 1
    
    if pos_theta:
        def tf(n, m):
            if n != m:
                return 1.j * (n - m) / (n - m)**2
            else:
                return sp.pi
    else:
        def tf(n, m):
            if n != m:
                return 2.j * ((n - m) * sp.pi * sp.cos((m - n) * sp.pi)
                              + sp.sin((m - n) * sp.pi)) / (2*sp.pi * (m - n)**2)
            else:
                return 0
            
    theta = sp.zeros((d, d), sp.complex128)
    for m in xrange(0, d):
        for n in xrange(0, d):
            theta[n, m] = tf(n, m)
    
    return theta
    
def U1_get_sqrtU(max_n=5):
    d = 2 * max_n + 1
    def f(n, m):
        return 1.j / (sp.pi * (0.5 + m - n))
        
    return sp.fromfunction(f, (d, d), dtype=sp.complex128)
    
def U1_get_interp_single(max_n=5):
    d = 2 * max_n + 1
    def f(n, j, m, k):
        if (m - n) + (k - j) + 1 == 0: 
            return 2. * (-1)**(m - n) / (1. + 2. * (m - n)) / sp.pi
        else:
            return 0
        
    res = sp.zeros((d, d, d, d), sp.complex128)
    for n in xrange(0, d):
        for m in xrange(0, d):
            for j in xrange(0, d):
                for k in xrange(0, d):
                    res[n, j, m, k] = f(n, j, m, k)
        
    return res.reshape(d**2, d**2) 
    
def U1_get_interp(ops, max_n=5):
    d = 2 * max_n + 1
    
    U_, P, P2, ReUUH, ReUUH_tp = ops
    U = U_[0][0]
    #sU = U1_get_sqrtU(max_n=max_n)
    
    UU = sp.kron(U, U)
    #sUsU = sp.kron(sU, sU)
    sUsU = U1_get_interp_single(max_n)
    
    UU_pow = [sp.eye(d**2)]
    for n in xrange(1, max_n + 1):
        UU_pow.append(UU.dot(UU_pow[-1]))
        
    def get_sU_pow(m):
        if m % 2 == 0:
            return UU_pow[m/2]
        else:
            return sUsU.dot(UU_pow[m/2])
    
    V = sp.zeros((d**3, d**3), dtype=sp.complex128)
    for n in xrange(-max_n, max_n + 1):
        m1 = d**2 * (n + max_n)
        m2 = m1 + d**2
        V[m1:m2, m1:m2] = (get_sU_pow(n) if n >= 0 else get_sU_pow(-n).conj().T)
    
    return V
    
def U1_get_mom_sup(ops, max_n=5):
    d = 2 * max_n + 1
    
    psi = 1./sp.sqrt(d) * sp.ones((d,), dtype=sp.complex128)
    
    return psi
    
def U1_get_mom_exp(ops, lam=1, max_n=5):
    psi = sp.exp(-lam * abs(sp.arange(-max_n, max_n + 1)))
    
    psi /= la.norm(psi)
    
    return psi
    
def U1_get_mom_test(ops, max_n=5):
    d = 2 * max_n + 1
    psi = sp.zeros((d,), dtype=sp.complex128)
    
    psi[max_n] = 1
    psi[max_n + 2] = psi[max_n - 2] = 0.1
    
    psi /= la.norm(psi)
    
    return psi
    
def apply_interp(V, psi, AA, trunc=None):
    d = AA.shape[0]
    D = AA.shape[2]
    
    V = V.reshape((d,d,d,d,d,d))
    
    #In V, the slowest varying index is the new site
    Vpsi = sp.tensordot(V, psi, axes=((3,), (0,))).copy()
    
    Vpsi = Vpsi.reshape((d**3, d**2))
    AA = AA.reshape((d**2, D**2))
    
    B123 = Vpsi.dot(AA)
    
    B123 = B123.reshape((d,d,d,D,D))
    
    #Place the virtual indices at the ends, move the new site to the middle
    #i.e. D, d, d*, d, D (d* is the new site)
    B123 = sp.transpose(B123, axes=(3, 1, 0, 2, 4))
    
    #Prepare for SVD splitting 12 and 3
    B12_3 = B123.reshape((D * d * d, d * D))
    
    B12_3_U, B12_3_s, B12_3_Vh = la.svd(B12_3, full_matrices=False)
    if not trunc is None:
        B12_3_s = B12_3_s[:trunc]
        B12_3_U = B12_3_U[:, :trunc]
        B12_3_Vh = B12_3_Vh[:trunc, :]
    
    B12_3_sr = la.diagsvd(sp.sqrt(B12_3_s), len(B12_3_s), len(B12_3_s))
    
    B3 = B12_3_sr.dot(B12_3_Vh).reshape((len(B12_3_sr), d, D))    
    B3 = sp.transpose(B3, axes=(1, 0, 2)).copy()
    
    B1_2 = B12_3_U.dot(B12_3_sr).reshape(D * d, d * B3.shape[1])
    
    B1_2_U, B1_2_s, B1_2_Vh = la.svd(B1_2, full_matrices=False)
    if not trunc is None:
        B1_2_s = B1_2_s[:trunc]
        B1_2_U = B1_2_U[:, :trunc]
        B1_2_Vh = B1_2_Vh[:trunc, :]
    
    B1_2_sr = la.diagsvd(sp.sqrt(B1_2_s), len(B1_2_s), len(B1_2_s))
    
    B1 = B1_2_U.dot(B1_2_sr).reshape(D, d, len(B1_2_sr))
    B1 = sp.transpose(B1, axes=(1, 0, 2))
    
    B2 = B1_2_sr.dot(B1_2_Vh).reshape(len(B1_2_sr), d, B3.shape[1])
    B2 = sp.transpose(B2, axes=(1, 0, 2))
    
    return B1, B2, B3
    
def get_interp_state(V, psi, s, trunc):
    assert s.L == 1
    s.update()
    AA = tdvp.tm.calc_AA(s.A[0], s.A[0])
    
    B1, B2, B3 = apply_interp(V, psi, AA, trunc=trunc)
    
    B31 = tdvp.tm.calc_AA(B3, B1)
    
    C3, C4, C1 = apply_interp(V, psi, B31, trunc=trunc)
    
    s2 = create_tdvp(trunc, s.ham, ham_tp=s.ham_tp, L=4)
    
    s2.A = [C1, B2, C3, C4]
    s2.update()
    
    #This gives the same result as get_interp_state_2 for |psi> = |0>. Good.
    #Also for |psi> = 1/sqrt(2) (|2> + |-2>).
    #Results should differ for |psi> = 1/sqrt(2) (|1> + |-1>).
    #..they do, but now get_interp_state changes the curvature
    #between the original sites... :( get_interp_state_2 does not.
    
    return s2
    
def get_interp_state_once(V, psi, s, trunc):
    assert s.L == 1
    s.update()
    AA = tdvp.tm.calc_AA(s.A[0], s.A[0])
    
    B1, B2, B3 = apply_interp(V, psi, AA, trunc=trunc)
    
    s2 = create_tdvp(trunc, s.ham, ham_tp=s.ham_tp, L=3)
    
    s2.A = [B1, B2, B3]
    s2.update()
    
    return s2
    
def get_interp_state_2(psi, s1, trunc=None, max_n=5, ops=None):
    if ops is None:
        ops = get_ops(1, max_n)
    
    d = 2 * max_n + 1
    D = s1.D
    
    psif = psi != 0
    d_eff = sp.count_nonzero(psif)
    ds = sp.arange(d)[psif]
    
    eyeD = sp.eye(D)
    
    B1 = sp.zeros((d, D, d_eff, D, d_eff), dtype=sp.complex128)
    for j in xrange(d_eff):
        B1[ds[j], :, j, :, j] = eyeD * psi[psif][j]
        
    B1 = B1.reshape((d, D * d_eff, D * d_eff))
    
    A = s1.A[0].copy()
    
    U = ops[0][0][0]
    U_pow = [sp.eye(d)]
    for n in xrange(1, max_n + 1):
        U_pow.append(U.dot(U_pow[-1]))
        
    sU = U1_get_sqrtU(max_n=max_n)
        
    sU_pow = [sp.eye(d)]
    for m in xrange(1, 2 * max_n + 1):
        if m % 2 == 0:
            sU_pow.append(U_pow[m/2])
        else:
            sU_pow.append(sU.dot(U_pow[m/2]))
    
    def get_sU_pow(m):
        if m < 0:
            return sU_pow[-m].conj().T
        else:
            return sU_pow[m]
            
    MPO = []
    for n in xrange(-max_n, max_n + 1):
        row = []
        if psif[n + max_n]:
            MPO.append(row)
            for m in xrange(-max_n, max_n + 1):
                if psif[m + max_n]:
                    row.append(get_sU_pow((m + n)))
            
    MPO = sp.array(MPO, dtype=sp.complex128)
                
    B2 = tdvp.tm.apply_MPO_local(MPO, A)
    
    s2 = create_tdvp(D * d_eff, s1.ham, ham_tp=s1.ham_tp, L=2)
    
    s2.A = [B1, B2]
    s2.update()
    
    return s2    
                
#def SU2_get_l(m): 
#    import sympy as sy
#    return sy.floor((-1 + sy.sqrt(1 + 8 * m)) / 2) / 2
#    
#def SU2_get_j(m):
#    return m - 2 * SU2_get_l(m)**2 - 2 * SU2_get_l(m)
#    
#def SU2_get_CG(m1, m2, m, itb):
#    import sympy.physics.quantum.cg as cg
#    return cg.CG(SU2_get_l(m1), SU2_get_j(m1), SU2_get_l(m2), SU2_get_j(m2), 
#                 SU2_get_l(m), SU2_get_j(m)).doit()

def SU2_get_Ws(tl):
    """This gets the generalized W states, from which irreps of group elements
    and the generators can be obtained.
    """
    from itertools import permutations
    def bool2int(x):
        y = 0
        for i,j in enumerate(x):
            y += j<<i
        return y
    
    W = [None] * (tl + 1)
    for m in xrange(tl + 1):
        perms = sp.array(list(set(permutations([0] * (tl - m) + [1] * m))))
        W[m] = sp.zeros((2**tl, ), dtype=sp.float64)
        for p in perms:
            W[m][bool2int(p)] += 1. / sp.sqrt(len(perms))
        #print W[m], sp.inner(W[m], W[m])
            
    return W
        

def SU2_get_irrep(U, tl, W=None):
    if W is None:
        W = SU2_get_Ws(tl)
    
    prod = sp.array([[1]])
    for i in xrange(tl):
        prod = sp.kron(U, prod)
        
    Ul = sp.zeros((tl + 1, tl + 1), dtype=sp.complex128)
    for m in xrange(tl + 1):
        for n in xrange(tl + 1):
            Ul[m, n] = W[m].T.dot(prod.dot(W[n]))
            
    return Ul
    
paus = [0.5 * sp.array([[0, 1], [1, 0]]), 
        0.5j * sp.array([[0, -1], [1, 0]]),
        0.5 * sp.array([[1, 0], [0, -1]])]
    
def SU2_get_gen(al, tl, W=None):
    if W is None:
        W = SU2_get_Ws(tl)
    
    pau = paus[al]
    
    M = sp.zeros((2**tl, 2**tl), dtype=sp.complex128)
    for n in xrange(tl):
        M += sp.kron(sp.eye(2**(n)), sp.kron(pau, sp.eye(2**(tl - n - 1))))
        
    tau = sp.zeros((tl + 1, tl + 1), dtype=sp.complex128)
    for m in xrange(tl + 1):
        for n in xrange(tl + 1):
            tau[m, n] = W[m].T.dot(M.dot(W[n]))
            
    return tau
    
def SU2_test_irreps(tl):
    l = tl / 2.
    W = SU2_get_Ws(tl)
    taus = [SU2_get_gen(al, tl, W=W) for al in [0, 1, 2]]
    eye_test = taus[0].dot(taus[0].conj().T) + taus[1].dot(taus[1].conj().T) + taus[2].dot(taus[2].conj().T)
    print "test generators:", sp.allclose(eye_test, sp.eye(tl + 1) * l * (l + 1))
    print "[t0,t1] - it2 = 0:", sp.allclose(taus[0].dot(taus[1]) - taus[1].dot(taus[0]), 1.j * taus[2])
    print "[t2,t0] - it1 = 0:", sp.allclose(taus[2].dot(taus[0]) - taus[0].dot(taus[2]), 1.j * taus[1])
    print "[t1,t2] - it0 = 0:", sp.allclose(taus[1].dot(taus[2]) - taus[2].dot(taus[1]), 1.j * taus[0])
    
    om = sp.rand(3)
    G_half = la.expm(1.j * (om[0] * paus[0] + om[1] * paus[1] + om[2] * paus[2]))
    print "G_half unitary", sp.allclose(G_half.dot(G_half.conj().T), sp.eye(2))
    Gl = la.expm(1.j * (om[0] * taus[0] + om[1] * taus[1] + om[2] * taus[2]))
    print "G_l unitary", sp.allclose(Gl.dot(Gl.conj().T), sp.eye(tl + 1))
    Gl_ = SU2_get_irrep(G_half, tl, W=W)
    print "G_l test", sp.allclose(Gl, Gl_)

def SU2_get_PL(max_2l=3):
    itb = SU2_build_index_ints(max_2l=max_2l)
    dim = len(itb)
        
    tl = 0
    tau_l = SU2_get_gen(0, tl)
    PL = [None] * 3
    for al in [0, 1, 2]:
        PL[al] = sp.zeros((dim, dim), dtype=sp.complex128)
        for mL in xrange(dim):
            for mR in xrange(dim):
                tlL, jpL, kpL = itb[mL]
                tlR, jpR, kpR = itb[mR]
                if not (tlL == tlR and kpL == kpR):
                    continue
                
                if tlL != tl:
                    tl = tlL
                    tau_l = SU2_get_gen(al, tl)

                PL[al][mL, mR] = tau_l[jpR, jpL]
            
    return PL
    
def SU2_get_PR(max_2l=3):
    itb = SU2_build_index_ints(max_2l=max_2l)
    dim = len(itb)
        
    tl = 0
    tau_l = SU2_get_gen(0, tl)
    PR = [None] * 3
    for al in [0, 1, 2]:
        PR[al] = sp.zeros((dim, dim), dtype=sp.complex128)
        for mL in xrange(dim):
            for mR in xrange(dim):
                tlL, jpL, kpL = itb[mL]
                tlR, jpR, kpR = itb[mR]
                if not (tlL == tlR and jpL == jpR):
                    continue
                
                if tlL != tl:
                    tl = tlL
                    tau_l = SU2_get_gen(al, tl)

                PR[al][mL, mR] = -tau_l[kpL, kpR]
            
    return PR
    
def SU2_test_U_PL(max_2l=3):
    PL = SU2_get_PL(max_2l=max_2l)
    U = SU2_get_U(max_2l=max_2l)
    
    print "U_0,0 = U*_1,1", sp.allclose(U[0][0], U[1][1].conj().T)
    print "U_0,1 = -U*_1,0", sp.allclose(U[0][1], -U[1][0].conj().T)
    #print "U_0,0 U*_0,0 = 1 - U_0,1 U*_0,1", sp.allclose(U[0][0].dot(U[1][1]), sp.eye(U[0][0].shape[0]) + U[0][1].dot(U[1][0]))
    
    for al in [0, 1, 2]:
        for m in [0, 1]:
            for n in [0, 1]:
                com = PL[al].dot(U[m][n]) - U[m][n].dot(PL[al])
                com_ = 0
                for k in [0, 1]:
                    com_ += paus[al][m, k] * U[k][n]
                print "[PL_%d, U_%d,%d] = (F_%d U)_%d,%d:" % (al, m, n, al, m, n), \
                      sp.allclose(com, com_), la.norm(com - com_)
    for al in [0, 1, 2]:
        for m in [0, 1]:
            for n in [0, 1]:
                com = PL[al].dot(U[m][n].conj().T) - U[m][n].conj().T.dot(PL[al])
                com_ = 0
                for k in [0, 1]:
                    com_ += -paus[al][k, m] * U[k][n].conj().T
                print "[PL_%d, U*_%d,%d] = (U*' F_%d)_%d,%d:" % (al, m, n, al, m, n), \
                      sp.allclose(com, com_), la.norm(com - com_)
                
    P2 = SU2_get_P2(max_2l=max_2l)
    P2_ = PL[0].dot(PL[0]) + PL[1].dot(PL[1]) + PL[2].dot(PL[2])
    print "P2 = PL_0^2 + PL_1^2 + PL_2^2:", sp.allclose(P2, P2_)
    
    d_maxtl = sp.sum((max_2l + 1)**2)
    start_maxtl = len(P2) - d_maxtl
    
    UUd = sp.zeros_like(U[0][0])
    for m in [0, 1]:
        for n in [0, 1]:
            UUd.fill(0)
            for k in [0, 1]:
                UUd += U[m][k].dot(U[n][k].conj().T)
            print "(U U^dag)_%d,%d = delta_%d,%d (restricted to all but highest irrep):" % (m, n, m, n), \
                  sp.allclose(UUd[:start_maxtl, :start_maxtl], 0 if m != n else sp.eye(start_maxtl))
            print "Error (norm distance) in highest irrep:", la.norm(UUd[start_maxtl:, start_maxtl:] - 0 if m != n else sp.eye(d_maxtl))
            
    eijk = sp.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    for al in [0, 1, 2]:
        for be in [0, 1, 2]:
            com = PL[al].dot(PL[be]) - PL[be].dot(PL[al])
            gas = sp.argwhere(eijk[al, be, :] != 0)
            if len(gas) > 0:
                ga = gas[0]
                com_ = -1.j * eijk[al, be, ga] * PL[ga]
            else:
                com_ = 0
            print "[PL_%d, PL_%d] = -1j * (eps^%d,%d_ga PL_ga)" % (al, be, al, be), sp.allclose(com, com_)

def SU2_get_L(U, max_2l=3):
    itb = SU2_build_index_ints(max_2l=max_2l)
    dim = len(itb)
        
    tl = 0
    G_l = sp.array([[1]])
    L = sp.zeros((dim, dim), dtype=sp.complex128)
    for mL in xrange(dim):
        for mR in xrange(dim):
            tlL, jpL, kpL = itb[mL]
            tlR, jpR, kpR = itb[mR]
            if not (tlL == tlR and kpL == kpR):
                continue
            
            if tlL != tl:
                tl = tlL
                G_l = SU2_get_irrep(U.conj().T, tl)

            L[mL, mR] = G_l[jpR, jpL]
        
    return L
    
def SU2_get_R(U, max_2l=3):
    itb = SU2_build_index_ints(max_2l=max_2l)
    dim = len(itb)
        
    tl = 0
    G_l = sp.array([[1]])
    R = sp.zeros((dim, dim), dtype=sp.complex128)
    for mL in xrange(dim):
        for mR in xrange(dim):
            tlL, jpL, kpL = itb[mL]
            tlR, jpR, kpR = itb[mR]
            if not (tlL == tlR and jpL == jpR):
                continue
            
            if tlL != tl:
                tl = tlL
                G_l = SU2_get_irrep(U, tl)

            R[mL, mR] = G_l[kpL, kpR]
        
    return R

def SU2_get_random_U():
    om = sp.rand(3)
    U = la.expm(1.j * (om[0] * paus[0] + om[1] * paus[1] + om[2] * paus[2]))
    return U

def SU2_test_LR(max_2l=3):
    U = SU2_get_random_U()
    V = SU2_get_random_U()

    L_U = SU2_get_L(U, max_2l=max_2l)
    R_V = SU2_get_R(V, max_2l=max_2l)
    print "[L_U, R_V] = 0:", sp.allclose(L_U.dot(R_V) - R_V.dot(L_U), 0), la.norm(L_U.dot(R_V) - R_V.dot(L_U))
    
    L_Ui = SU2_get_L(U.conj().T, max_2l=max_2l)
    R_Vi = SU2_get_R(V.conj().T, max_2l=max_2l)
    print "L_Ui = L_U^dag:", sp.allclose(L_U.conj().T, L_Ui)
    print "R_Vi = R_V^dag:", sp.allclose(R_V.conj().T, R_Vi)
    
    L_I = SU2_get_L(sp.eye(2), max_2l=max_2l)
    R_I = SU2_get_L(sp.eye(2), max_2l=max_2l)
    print "L_I = I:", sp.allclose(L_I, sp.eye(len(R_I)))
    print "R_I = I:", sp.allclose(R_I, sp.eye(len(R_I)))

def SU2_get_P2(max_2l=3):
    twols = sp.arange(max_2l + 1)
    dim = sp.sum((twols + 1)**2)
    
    P2diag = []
    for twol in twols:
        P2diag += [twol/2. * (twol/2. + 1)] * (twol + 1)**2
    
    assert len(P2diag) == dim
    
    return sp.diag(sp.array(P2diag, dtype=sp.float64))
    
def SU2_build_index(max_2l=3):
    tbl = []
    for twol in xrange(max_2l + 1):
        l = twol/2.
        tbl += [[l, jp - l, kp - l] for jp in xrange(twol + 1) for kp in xrange(twol + 1)]
    return sp.array(tbl, dtype=sp.float64)
    
def SU2_build_index_ints(max_2l=3):
    tbl = []
    for twol in xrange(max_2l + 1):
        tbl += [[twol, jp, kp] for jp in xrange(twol + 1) for kp in xrange(twol + 1)]
    return sp.array(tbl)
    
def SU2_build_CGs(max_2l=3):
    """This grabs a tensor of Clebsch-Gordan coefficients <lL,mL|lR,mR;1/2,mM>
       skipping the zeros where the l1 != l3 +/- 1/2.
       There is a cutoff in l given by max_2l.
       
       Uses sympy to get the CG coeffients exactly before converting to floats.
    """
    from sympy.physics.quantum.cg import CG
    
    vtb = []
    half = sy.S(1) / 2
    for twolL in xrange(max_2l + 1):
        vtb_ = [None] * (max_2l + 1)
        lL = sy.S(twolL) / 2
        for twolR in [twolL - 1, twolL + 1]:
            if twolR > max_2l or twolR < 0:
                continue
            lR = sy.S(twolR) / 2
            vtb_[twolR] = [[[sy.N(CG(lR, mRp - lR, half, mMp - half, lL, mLp - lL).doit())
                            for mRp in xrange(twolR + 1)]
                            for mMp in [0, 1]]
                            for mLp in xrange(twolL + 1)]
        vtb.append(vtb_)
    return vtb
    
def SU2_test_CGs(max_2l=3):
    CGs = SU2_build_CGs(max_2l=max_2l)
    
    UCG = sp.zeros((6, 6), dtype=sp.float64)
    UCG.fill(sp.NaN)
    tlR = 2
    for tlL in [tlR - 1, tlR + 1]:
        if tlL < 0 or tlL > max_2l:
            continue
        shft = (4 if tlL == 1 else 0)
        for mLp in xrange(tlL + 1):
            for mMp in [0, 1]:
                for mRp in xrange(tlR + 1):
                    UCG[shft + mLp, mMp * (tlR + 1) + mRp] = CGs[tlL][tlR][mLp][mMp][mRp]
    print UCG
    print "CGd CG = 1:", sp.allclose(UCG.dot(UCG.conj().T), sp.eye(len(UCG)))
    
def SU2_test_trunc_CG(max_2l=3):
    CGs = SU2_build_CGs(max_2l=max_2l)
    dim = sp.sum(sp.arange(max_2l + 1) + 1)
    
    def get_M(tl, m):
        return (tl**2 + tl + 2 * m) / 2
        
    #print get_M(0, 0), get_M(1, 0), get_M(1, 1), get_M(2, 0), get_M(2, 1), get_M(2, 2), get_M(3, 0), get_M(3, 1)
    
    eye_ = sp.zeros((dim, dim), dtype=sp.float64)
    eye_.fill(sp.NaN)
    for tl1 in xrange(max_2l + 1):
        for tl2 in xrange(max_2l + 1):
            for m1 in xrange(tl1 + 1):
                for m2 in xrange(tl2 + 1):
                    entry = 0
                    for tl in xrange(max_2l + 1):
                        for m in xrange(tl + 1):
                            for mh in [0, 1]:
                                if CGs[tl1][tl] is None or CGs[tl2][tl] is None:
                                    continue
                                entry += (tl + 1) / sp.sqrt(tl1 + 1) / sp.sqrt(tl2 + 1) * CGs[tl1][tl][m1][mh][m] * CGs[tl2][tl][m2][mh][m]
                    eye_[get_M(tl1, m1), get_M(tl2, m2)] = entry
    return eye_.diagonal()
    
def SU2_get_U(max_2l=3):
    itb = SU2_build_index_ints(max_2l=max_2l)
    dim = len(itb)
    
    CGs = SU2_build_CGs(max_2l=max_2l)

    U = [[None, None], [None, None]]
    for m in range(2):
        for n in range(2):
            U[m][n] = sp.zeros((dim, dim), dtype=sp.float64)
            for mL in xrange(dim):
                for mR in xrange(dim):
                    tlL, jpL, kpL = itb[mL]
                    tlR, jpR, kpR = itb[mR]
                    
                    if CGs[tlL][tlR] is None: #Combination is always zero.
                        continue
                    
                    CGj = CGs[tlL][tlR][jpL][m][jpR]
                    CGk = CGs[tlL][tlR][kpL][n][kpR]
                    U[m][n][mL, mR] = (sp.sqrt((tlR + 1.) / (tlL + 1.)) * CGj * CGk)
    return U
    
def SU2_get_RetrUUH(U):
    d = U[0][0].shape[0]
    trUUH = sp.zeros((d**2, d**2), dtype=U[0][0].dtype)
    for j in [0, 1]:
        for k in [0, 1]:
            trUUH += 0.5 * (sp.kron(U[j][k], U[j][k].T) + sp.kron(U[j][k].T, U[j][k]))
            
    return trUUH.reshape(d, d, d, d)
    
def SU2_get_RetrUUH_tp(U):
    trUUH = []
    for k in [0, 1]: #abbreviated version, exploiting unitary structure
        trUUH.append([U[0][k], U[0][k].T])
        trUUH.append([U[0][k].T, U[0][k]])
            
    return trUUH
    
def SU2_get_ops(max_2l=3):
    PL = SU2_get_PL(max_2l=max_2l)
    P2 = SU2_get_P2(max_2l=max_2l)
    U = SU2_get_U(max_2l=max_2l)
    RetrUUH = SU2_get_RetrUUH(U)
    RetrUUH_tp = SU2_get_RetrUUH_tp(U)
    
    return U, PL, P2, RetrUUH, RetrUUH_tp
    
def SU2_test_GI(max_2l=3):
    U, PL, P2, RetrUUH, RetrUUH_tp = SU2_get_ops(max_2l=max_2l)
    
    RetrUUH = RetrUUH.reshape((len(P2)**2, len(P2)**2))
    
    om = sp.rand(3)
    U = la.expm(1.j * (om[0] * paus[0] + om[1] * paus[1] + om[2] * paus[2]))

    L_U = SU2_get_L(U, max_2l=max_2l)
    R_U = SU2_get_R(U, max_2l=max_2l)
    
    LR_U = L_U.dot(R_U)
    LR_U_12 = sp.kron(LR_U, LR_U)

    print "Gauge noninvariance:"
    print "LR_U U_00 LR_U* != U_00:", not sp.allclose(LR_U.dot(U[0][0]).dot(LR_U.conj().T), U[0][0])
    print "LR_U U_01 LR_U* != U_01:", not sp.allclose(LR_U.dot(U[0][1]).dot(LR_U.conj().T), U[0][1])
    print "LR_U U_10 LR_U* != U_10:", not sp.allclose(LR_U.dot(U[1][0]).dot(LR_U.conj().T), U[1][0])
    print "LR_U U_11 LR_U* != U_11:", not sp.allclose(LR_U.dot(U[1][1]).dot(LR_U.conj().T), U[1][1])
    print "LR_U PL_0 LR_U* != PL_0:", not sp.allclose(LR_U.dot(PL[0]).dot(LR_U.conj().T), PL[0])
    print "LR_U PL_1 LR_U* != PL_1:", not sp.allclose(LR_U.dot(PL[1]).dot(LR_U.conj().T), PL[1])
    print "LR_U PL_2 LR_U* != PL_2:", not sp.allclose(LR_U.dot(PL[2]).dot(LR_U.conj().T), PL[2])
    
    print "\nGauge invariance (global rotation from left and right):"
    print "LR_U P2 LR_U* = P2:", sp.allclose(LR_U.dot(P2).dot(LR_U.conj().T), P2)
    print "(LR_U x LR_U) RetrUUH (LR_U* x LR_U*) = RetrUUH:", sp.allclose(LR_U_12.dot(RetrUUH).dot(LR_U_12.conj().T), RetrUUH)
    
    L_U_12 = sp.kron(L_U, L_U)
    R_U_12 = sp.kron(R_U, R_U)
    
    print "\nRotation invariance:"
    print "L_U P2 L_U* = P2:", sp.allclose(L_U.dot(P2).dot(L_U.conj().T), P2)
    print "R_U P2 R_U* = P2:", sp.allclose(R_U.dot(P2).dot(R_U.conj().T), P2)
    print "(L_U x L_U) RetrUUH (L_U* x L_U*) = RetrUUH:", sp.allclose(L_U_12.dot(RetrUUH).dot(L_U_12.conj().T), RetrUUH)
    print "(R_U x R_U) RetrUUH (R_U* x R_U*) = RetrUUH:", sp.allclose(R_U_12.dot(RetrUUH).dot(R_U_12.conj().T), RetrUUH)
    
def SU2_test_rotor(max_2l=3):
    U, PL, P2, RetrUUH, RetrUUH_tp = SU2_get_ops(max_2l=max_2l)
    
    Fs = paus + [0.5j * sp.eye(2)]
    N = [sum([-1.j * Fs[al][m, n] * U[n][m] for m in [0, 1] for n in [0, 1]])
                                                      for al in [0, 1, 2, -1]]
    
    NN = sum([sp.kron(Nal, Nal) for Nal in N])
    
    print "NN = NN*", sp.allclose(NN, NN.conj().T)
    
    d_maxtl = sp.sum((max_2l + 1)**2)
    start_maxtl = len(P2) - d_maxtl
    for al in [0, 1, 2, -1]:
        for be in [0, 1, 2, -1]:
            com = (N[al].dot(N[be]) - N[be].dot(N[al]))[:start_maxtl, :start_maxtl]
            print "[N_%d, N_%d] = 0 (up to last irrep)" % (al, be), sp.allclose(com, 0)
    
    print "RetrUUH = 2 * NN", sp.allclose(RetrUUH.reshape(NN.shape), 2 * NN)
    
    eijk = sp.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    
    for al in [0, 1, 2]:
        for be in [0, 1, 2, -1]:
            com = PL[al].dot(N[be]) - N[be].dot(PL[al])
            ga_ = None
            if be == -1:
                ga_ = al
                c = -1
            elif al == be:
                ga_ = -1
                c = 1
            else:
                ga_ = sp.argwhere(eijk[al, be, :] != 0)[0]
                c = eijk[al, be, ga_]
            
            com_ = -0.5j * c * N[ga_]
            #print la.norm(com), la.norm(com_)
            print "[PL_%d, N_%d] = -0.5j * %d * N_%d" % (al, be, c, ga_), sp.allclose(com, com_)
                                       
def get_ham(g2inv, ops, a=1.):
    U, P, P2, ReUUH, ReUUH_tp = ops
    d = P2.shape[0]
    
    if g2inv == 0:
        h = (1 / (2. * a) * sp.kron(P2, sp.eye(d))).reshape(d, d, d, d)
    else:
        h = (1 / (2. * a * g2inv) * sp.kron(P2, sp.eye(d)).reshape(d, d, d, d) 
             - 2 * g2inv / a * ReUUH)
    return h
    
def get_ham_tp(g2inv, ops, a=1.):
    U, P, P2, ReUUH, ReUUH_tp = ops
    d = P2.shape[0]
    
    if g2inv == 0:
        h = [[1 / (2. * a) * P2, sp.eye(d)]]
    else:
        fac = -2. * g2inv / a
        ReUUH_ = [[fac * tp[j] if j == 0 else tp[j] for j in [0, 1]] for tp in ReUUH_tp]
        h = [[1 / (2. * a * g2inv) * P2, sp.eye(d)]] + ReUUH_
    return h

def create_tdvp(D, ham, L=1, zero_tol=0, sanity_checks=False, ham_tp=None):
    s = tdvp.EvoMPS_TDVP_Uniform(D, ham.shape[0], ham, L=L)
    s.zero_tol = zero_tol
    s.sanity_checks = sanity_checks
    s.itr_atol = 1E-14
    s.itr_rtol = 1E-14
    s.ev_arpack_CUDA = use_CUDA
    s.PPinv_use_CUDA = use_CUDA
    s.ham_tp = ham_tp
    return s

def state_file(s, G, g2inv, max_ir, max_tries=10, loc='state_data/'):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    for i in xrange(max_tries):
        grnd_fname = "ke_state_G%d_L%d_D%d_maxir%d_g2inv%.4f_T%s_%u.npy" % (
                                           G, s.L, s.D, max_ir, g2inv, ts, i)
        try:
            f = open(loc + grnd_fname, 'rb')
            f.close()
        except IOError:
            f = open(loc + grnd_fname, 'wb')
        break
        
    return f, grnd_fname

def save_result(s, G, g2inv, max_ir, wv=sp.NaN, gap_wv=sp.NaN, v0=None, ops=None, 
                existing_fn=None, res_file='petal_res.txt'):
    exc_fn = ''
    if existing_fn is None:
        f, fn = state_file(s, G, g2inv, max_ir)
        s.save_state(f, userdata={'g2inv': g2inv, 'eta': s.eta})
        if not v0 is None:
            exc_fn = fn[:-4] + '_v0.npy'
            sp.save(exc_fn, v0)
    else:
        fn = existing_fn
    
    row = get_res_row(s, G, g2inv, max_ir, wv=wv, gap_wv=gap_wv, ops=ops)
    
    row[cols['exc_fn']] = exc_fn
    row[cols['fn']] = fn
    
    assert len(filter(lambda x: x is None, row)) == 0
    
    resf = open(res_file, 'a')
    fcntl.flock(resf, fcntl.LOCK_EX)
    resf.write("\t".join(map(str, row)) + "\n")
    resf.close()
    
    return row

def get_res_row(s, G, g2inv, max_ir, ops=None, wv=sp.NaN, gap_wv=sp.NaN):
    d = s.q
    
    U, P, P2, ReUUH, ReUUH_tp = ops
    
    exUs = sp.array(map(lambda k: s.expect_1s(U[0][0], k=k), range(s.L)))
    exPs = sp.array(map(lambda k: s.expect_1s(P[0], k=k), range(s.L)))
    
    U2 = U[0][0].dot(U[0][0])
    exU2s = sp.array(map(lambda k: s.expect_1s(U2, k=k), range(s.L)))
    
    exP2s = sp.array(map(lambda k: s.expect_1s(P2, k=k).real, range(s.L)))
    
    exReUUHs = sp.array(map(lambda k: s.expect_2s_tp(ReUUH_tp, k=k).real, range(s.L)))
    
    ReUUH2 = ReUUH.reshape(d**2, d**2).dot(ReUUH.reshape(d**2, d**2)).reshape(d,d,d,d)
    s.calc_AA()
    exReUUH2s = sp.array(map(lambda k: s.expect_2s(ReUUH2, k=k).real, range(s.L)))
    
    entrs = sp.array(map(s.entropy, range(s.L)))
    
    row = [None] * len(cols)
    row[cols['G']] = G
    row[cols['g2inv']] = g2inv
    row[cols['D']] = s.D
    row[cols['L']] = s.L
    row[cols['max_ir']] = max_ir
    row[cols['eta']] = s.eta.real
    row[cols['max_ir_oc']] = s.basis_occupancy()[-1]
    row[cols['energy']] = s.h_expect.real
    row[cols['cl']] = s.correlation_length()
    row[cols['entr_max']] = entrs.max()
    row[cols['U_av']] = exUs.mean()
    row[cols['U_0']] = exUs[0]
    row[cols['U2_0']] = exU2s[0]
    row[cols['P_av']] = exPs.mean()
    row[cols['P_0']] = exPs[0]
    row[cols['P2_0']] = exP2s[0]
    row[cols['ReUUH_av']] = exReUUHs.mean()
    row[cols['ReUUH_0']] = exReUUHs[0]
    row[cols['ReUUH2_0']] = exReUUH2s[0]
    row[cols['wv_dom']] = wv
    row[cols['wv_fit']] = sp.NaN
    row[cols['gap']] = gap_wv
    
    return row
    
DCOLRANGE = sp.arange(16, 256 + 1)
DCOLPAL = cm.jet
def get_D_colmap():
    cmap = cm.ScalarMappable(cmap=DCOLPAL)
    cmap.set_array(DCOLRANGE)
    cmap.autoscale()

    return cmap

LMARKERS = [None, '+', 'x', 'x', 'x']
def get_markers():
    #import matplotlib.markers as mrk
    #return mrk.MarkerStyle.markers.keys()
    return LMARKERS
    
def plot_colormap():
    import matplotlib.pyplot as plt
    
    cm = get_D_colmap()
    
    Ds = range(16, 128)
    
    plt.scatter(Ds, Ds, s=80, c=cm.to_rgba(Ds))
    plt.show()

def plot_Ds(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols), dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], ress[mask, cols['D']], s=80, c=cm.to_rgba(ress[mask, 2]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$D$')
    plt.show()
    
def plot_energy(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols), dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], ress[mask, cols['energy']], s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$h$')
    plt.show()
    
def plot_energy_deriv(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols), dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    x = ress[:, cols[PARAM]]
    d1_y_Ham = -2. * ress[:, cols['ReUUH_av']] - x**(-2) / 2. * ress[:, cols['P2_0']]
    
    dx = sp.ediff1d(x)
    d_x = x[:-1] + dx / 2.
    d1_y = sp.ediff1d(ress[:, cols['energy']]) / dx
    d2_y = sp.ediff1d(d1_y_Ham) / dx
    dx_fd = dx[:-1]
    d2_x_fd = x[:-2] + sp.ediff1d(d_x)
    d2_y_fd = sp.ediff1d(d1_y) / dx_fd
    
    plt.figure(1)
    plt.plot(d_x, d1_y, '--')
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(x[mask], d1_y_Ham[mask], s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$dE/d' + PARAM_TEX + '$')
    
    plt.figure(2)
    plt.plot(d_x, d2_y, '-o')
    plt.plot(d2_x_fd, d2_y_fd, '--')
    plt.xlabel('$g^{-2}$')
    plt.ylabel('$d^2E/d(' + PARAM_TEX + ')^2$')
    print "d2e min at:", d_x[d2_y.argmin()]
    plt.show()
    
def plot_beta(res_file="petal_res.txt", G=1, fmt='o', lbl=None, plot_an=True, 
              usualg=False, SBtol=1E-4, ecol=None, fcol=None, **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols), dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, G=G, **kwargs)
    
    filt = abs(ress[:, cols['U_0']]) < SBtol
    ress = ress[filt, :]
    
    if G == 1:
        N = 2
        g2i_fac = sp.sqrt(2)
        E_fac = 1. / sp.sqrt(2)
    elif G == 2:
        N = 4
        g2i_fac = 4
        E_fac = 1.
    
    x = 2 * (g2i_fac * ress[:, cols[PARAM]])**2
    #plot_x = 1. / (g2i_fac * ress[:, cols[PARAM]])
    #d_plot_x = sp.ediff1d(plot_x)
    #plot_x = plot_x[:-1] - d_plot_x / 2
    #plot_x = plot_x[:-1]
    y = 2 * g2i_fac * ress[:, cols[PARAM]] * E_fac * ress[:, cols['gap']]

    dx = sp.ediff1d(x)
    d_x = x[:-1] + dx / 2    #use midpoints as x coord for the derivative
    plot_x = sp.sqrt(2 / d_x)
    dy = sp.ediff1d(y)
    d1_y = dy / dx
    d_y = y[:-1] + dy / 2
    
    m_beta_g = 1. / (1 - 2 * d_x * d1_y / d_y)
    
    g_app = sp.linspace(0, 4, num=1000)
    
    #The following strong coupling series expansions are from Hamer et al. 1979
    if G == 2:
        _p = [1, 0.064665, -0.19101, 0.015944, 0.052396, -7.2098E-3, 5.572E-5, 1.2339E-3]
        _p = _p[::-1]
        _q = [1, 0.064665, 0.47565, 0.059053, 0.063937, 0.0124, 0.02263, -4.419E-4, 3.3764E-3]
        _q = _q[::-1]
        y_app = sp.polyval(_p, 1./g_app) / sp.polyval(_q, 1./g_app)
    elif G == 1:
        _p = [1., -0.1413, -0.2589, -0.1662, 0.09704]
        _p = _p[::-1]
        _q = [1., 1.8587, 0.9584, 0.1664, -0.07654]
        _q = _q[::-1]
        y_app = sp.polyval(_p, 2./g_app**2) / sp.polyval(_q, 2./g_app**2)

    y_weak = (N - 2) * g_app / (2 * sp.pi) + (N - 2) * g_app**2 / (4 * sp.pi**2)
    
    xfac = 1.
    yfac = 1.
    if usualg:
        xfac = g2i_fac
        yfac = 1. / g2i_fac
    
    if plot_an:
        plt.plot([0, xfac * 4], [yfac, yfac], 'k-')
        if N > 2:
            plt.plot(xfac * g_app, yfac * y_weak, 'g-', label='WC')
        if G == 1:
            filt = y_app < 0
            app_start = g_app[filt].max()
            filt = g_app > app_start
        else:
            filt = sp.ones((g_app.shape[0],), dtype=bool)
        l, = plt.plot(xfac * g_app[filt], yfac * y_app[filt], 'k--', label="Pad\\'{e}")
        l.set_dashes((1.5, 0.5))
        
    filt = (m_beta_g > 0) * (m_beta_g < 1)
    plt.plot(xfac * plot_x[filt], yfac * m_beta_g[filt], fmt, label=lbl,
             markersize=2.5, markeredgewidth=0.5,
             markeredgecolor=ecol, markerfacecolor=fcol)
#    pf, cf = sp.polyfit(xfac * plot_x[filt], yfac * m_beta_g[filt], 1, cov=True)
#    print "fit", pf, cf, -pf[1]/pf[0]
#    plt.plot(xfac * g_app, sp.polyval(pf, xfac * g_app), '-')
    
    if usualg:
        plt.xlabel('$g^2$' )
        plt.ylabel('$-\\beta(g^2) / g^{2}$')
    else:
        plt.xlabel('$\\tilde{g}$' )
        plt.ylabel('$-\\beta(\\tilde g) / \\tilde{g}$')
    #plt.xlim((0, 2.1))
    #plt.ylim((0, 1.2))
    plt.show()
    
def filt_res(ress, G=1, pars=None, max_irs=None, Ds=None, Ls=None, lowest_en=False, par_decimals=4):
    filt = ress[:, cols['G']] == G
    ress = ress[filt, :]
    
    if not max_irs is None:
        filt = sp.in1d(ress[:, cols['max_ir']], max_irs)
        ress = ress[filt, :]
    
    if not Ds is None:
        filt = sp.in1d(ress[:, cols['D']], Ds)
        ress = ress[filt, :]
        
    if not pars is None:
        filt = sp.in1d(sp.around(ress[:, cols[PARAM]], decimals=par_decimals), sp.around(pars, decimals=par_decimals))
        ress = ress[filt, :]
    
    if Ls is None:
        Ls = sp.unique(ress[:, cols['L']])
    else:
        filt = sp.in1d(ress[:, cols['L']], Ls)
        ress = ress[filt, :]
        
    if lowest_en:
        ka_rounded = sp.around(ress[:, cols[PARAM]], decimals=par_decimals)
        en_rounded = sp.around(ress[:, cols['energy']], decimals=12)

        to_sort = sp.column_stack((ka_rounded, ress[:, cols[PARAM]], en_rounded,
                                   ress[:, cols['eta']])).view('complex128,complex128,complex128,complex128')

        sort1 = sp.argsort(to_sort, axis=0, order=['f1']) #first sort by full precision param
        #must now sort by rounded values for unique to work properly
        #must also use a stable algo to maintain ordering
        sort2 = sp.argsort(to_sort[sort1], order=['f0', 'f2', 'f3'], axis=0, kind='mergesort')

        sort_final = sort1[sort2]
        to_sort = to_sort.view(sp.float64)
        to_sort = sp.squeeze(to_sort)
        sort_final = sp.squeeze(sort_final)
        
        #filter based on rounded values. return_index forces merge_sort, which is stable!
        blah, unq_ind, blah2 = sp.unique(to_sort[sort_final, 0], return_index=True, return_inverse=True)
        sort_filt = sort_final[unq_ind]
        
        ress = ress[sort_filt, :]
    else:
        sort1 = sp.argsort(ress[:, cols[PARAM]])
        ress = ress[sort1, :]
        
    return ress, Ls
    
def get_num_ress(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
                         
    ress, Ls = filt_res(ress, **kwargs)
    
    return ress
    
def get_col(colname, res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
                         
    ress, Ls = filt_res(ress, **kwargs)
    
    return ress[:, cols[colname]]
    
def plot_col(colname, res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], ress[mask, cols[colname]], s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel(colname)
    plt.show()
    
def plot_U00(res_file="petal_res.txt", broken_tol=1E-3, **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]].real, abs(ress[mask, cols['U_0']]), s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
        broken = abs(ress[mask, cols['U_0']]) > broken_tol
        print "L=", L, "last symm. g2inv =", ress[mask, cols[PARAM]][broken].min().real
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$U_{0,0}$')
    plt.show()
    
def plot_gap(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        if not sp.all(sp.isnan(ress[mask, cols['gap']])):
            #print ress[mask, 1], ress[mask, -1], cm.to_rgba(ress[mask, 2])
            plt.scatter(ress[mask, cols[PARAM]], ress[mask, cols['gap']], s=80, 
                        c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$(E_1 - E_0)$')
    plt.show()
    
def plot_gap_log(res_file="petal_res.txt", G=1, plot_an=True, lbl=None, 
                 fmt='o', ecol=None, fcol=None, usualg=False, SBtol=1E-3, 
                 C=None, eta_corr=True, **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()

    if G == 1:
        N = 2
        g2i_fac = sp.sqrt(2)
        E_fac = 1. / sp.sqrt(2)
    elif G == 2:
        N = 4
        g2i_fac = 4
        E_fac = 1.
    
    x_app = sp.linspace(0, 3, num=1000) #1/g'
    
    #The following strong coupling series expansions are from Hamer et al. 1979
    if G == 2:
        #se = [1.0, -1./6, +0.005208, +0.0003798, +0.00000417397, -0.00000036076, -0.000000252236]
        #The above is just 1/3 times the following:
        se = [3., -0.5, 0.015625, 1.139323E-3, 1.25219E-5, -1.082289E-6, -7.56709E-7]
        se = se[::-1]
        y_app = sp.sqrt(1./x_app**3 * 1./4.) * sp.polyval(se, 2 * x_app**2)
    elif G == 1:
        se = [1., -1., 0.125, 0.03125, 1.438802E-2, 6.002063E-3, 2.26148E-4, 6.95799E-4, -1.752E-4]
        se = se[::-1]
        y_app = sp.sqrt(1./x_app**3 * 1./4.) * sp.polyval(se, 2 * x_app**2)
        ##Hornby and Barber 1985
        #se = [1., -2., 0.5, 0.25, 0.2302083, 0.192065972, 0.0144735794,
        #      0.0890622894, -0.0448071196, 0.0359987647, -0.0818017597]
        #se = se[::-1]
        #y_app = sp.sqrt(1./x_app**3 * 1./4.) * sp.polyval(se, 1 * x_app**2)
    
    xfac = 1
    yfac = 1
    if usualg:
        xfac = 1. / g2i_fac
        yfac = sp.sqrt(g2i_fac) / E_fac
    
    if plot_an:
        l, = plt.plot(xfac * x_app, yfac * y_app, 'k--', label="SC")
        l.set_dashes((1.5, 0.5))
    
    if G == 2 and plot_an:
        #value from Hasenratz et al. 1990, modified for spatial discretisation 
        #using Shigemitsu & Kogut 1981 (based on Parisi's result for the
        #Euclidean lattice). 
        if C is None:
            C = 8 * sp.exp(0.5) * sp.sqrt(32./(sp.pi * sp.e))
        y_weak = C * sp.sqrt(sp.pi) * sp.exp(-sp.pi * x_app)
        #Correct for time-space asymmetry s.t. renormalised theory is LI.
        #Also from Shigemitsu & Kogut.
        if eta_corr:
            # Comparing with the paper, this original line
            # y_weak /= sp.sqrt(1 + 1. / x_app / sp.pi)
            # had a sign error!
            y_weak /= sp.sqrt(1 - 1. / x_app / sp.pi)
        plt.plot(xfac * x_app, yfac * y_weak, 'g-', label='WC')
        
    ress, Ls = filt_res(ress, G=G, **kwargs)
    filt = abs(ress[:, cols['U_0']]) < SBtol
    ress = ress[filt, :]    
    
    plt.plot(xfac * g2i_fac * ress[:, cols[PARAM]], 
                yfac * E_fac * ress[:, cols['gap']] / sp.sqrt(g2i_fac * ress[:, cols[PARAM]]), 
                fmt, label=lbl, markersize=2.5, markeredgewidth=0.5,
                markeredgecolor=ecol, markerfacecolor=fcol)
    
    if usualg:
        plt.xlabel('$g^{-2}$' )
        plt.ylabel('$(E_1 - E_0) g / \\sqrt{\\eta}$')
    else:
        plt.xlabel('$1 / \\tilde g$' )
        plt.ylabel('$(E_1 - E_0) \\sqrt{\\tilde g} / \\sqrt{\\eta}$')
    plt.yscale('log')
    #plt.xlim((0, 3))
    #plt.ylim((0.03, 10))
    plt.show()
    
def plot_wv(res_file="petal_res.txt", scattersize=80, lw=None, pos_shift=False, 
            return_data=False, **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    plts = []
    
    allxs = []
    allys = []
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        if sp.sometrue(mask):
            notlocks = (abs(ress[mask, cols['wv_dom']] / L) % (sp.pi / 4)) >= 1E-6
            print "Last not-lock at:", ress[mask, cols[PARAM]][notlocks].max()
        if L == 1:
            xs = ress[mask, cols[PARAM]]
            ys = abs(ress[mask, cols['wv_dom']]) / sp.pi
            cs = ress[mask, cols['D']]
        elif L > 1:
            xs = []
            ys = []
            cs = []
            maxn = int(L) / 2
            for n in xrange(maxn + 1):
                if n > 0:
                    xs.append(ress[mask, cols[PARAM]])
                    ys.append(n * 2. / L - abs(ress[mask, cols['wv_dom']]) / sp.pi / L)
                    cs.append(ress[mask, cols['D']])
                if pos_shift and n < maxn:
                    xs.append(ress[mask, cols[PARAM]])
                    ys.append(n * 2. / L + abs(ress[mask, cols['wv_dom']]) / sp.pi / L)
                    cs.append(ress[mask, cols['D']])
            xs = sp.array(xs).ravel()
            ys = sp.array(ys).ravel()
            cs = sp.array(cs).ravel()
        res = plt.scatter(xs, ys, s=scattersize, c=cm.to_rgba(cs), marker=mrks[int(L)],
                          lw=lw)
        plts.append(res)
        allxs.append(xs)
        allys.append(ys)
    allxs = sp.concatenate(allxs)
    allys = sp.concatenate(allys)
    tbl = sp.column_stack((allxs, allys))

    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$k/\\pi$')
    plt.show()
    
    return plts, tbl
    
def plot_wv_locks(Ds=None, res_file="petal_res.txt", scattersize=80, lowest_en=True, **kwargs):
    ress_ = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    if Ds is None:
        Ds = sp.unique(ress_[:, cols['D']])
    else:
        Ds = sp.array(Ds)
    
    ka_lls = sp.zeros((len(Ds),), dtype=sp.float64)
    ka_lls.fill(sp.NaN)
    for i, D in enumerate(Ds):
        ress, Ls = filt_res(ress_.copy(), Ds=[D], lowest_en=lowest_en, **kwargs)
        notlocks = (abs(ress[:, cols['wv']] / ress[:, cols['L']]) % (sp.pi / 4)) >= 1E-6
        ka_ll = ress[:, cols[PARAM]][notlocks].max()
        ka_lls[i] = ka_ll.real
                
    pfit, covfit = sp.polyfit(1. / Ds, ka_lls, 1, cov=True)
    fitka = sp.linspace(0, (1. / Ds).max() * 1.1)
    
    print pfit[1], sp.sqrt(covfit[1, 1])
    
    plt.plot(1. / Ds, ka_lls, 'bo')
    plt.plot(fitka, sp.polyval(pfit, fitka), 'k-')
    plt.errorbar(0, pfit[1], yerr=sp.sqrt(covfit[1, 1]), fmt='b')
    plt.xlabel('$1/D$')
    plt.ylabel('$' + PARAM_TEX + '_\\mathrm{lock}$')
    plt.xlim((-0.0005, 0.015))
    plt.show()
    
def plot_lcl(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], sp.log(ress[mask, cols['cl']]), s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$\\log(\\xi)$')
    plt.show()
    
def plot_icl(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], 1/ress[mask, cols['cl']], s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$\\xi^{-1}$')
    plt.show()
    
def plot_entropy(res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
    
    import matplotlib.pyplot as plt
    
    mrks = get_markers()
    cm = get_D_colmap()
    
    ress, Ls = filt_res(ress, **kwargs)
    
    for L in Ls:
        mask = (ress[:, cols['L']] == L)
        plt.scatter(ress[mask, cols[PARAM]], ress[mask, cols['entr_max']], s=80, 
                    c=cm.to_rgba(ress[mask, cols['D']]), marker=mrks[int(L)])
    plt.xlabel('$' + PARAM_TEX + '$' )
    plt.ylabel('$S$')
    plt.show()
    
def plot_cc(Ds, L=1, max_ir=5, pars=None, offset=0, res_file="petal_res.txt", **kwargs):
    ress = sp.genfromtxt(res_file, delimiter="\t", usecols=range(num_floatcols),
                         dtype=sp.complex128)
                         
    import matplotlib.pyplot as plt
    
    ress, Ls = filt_res(ress, pars=pars, Ds=Ds, Ls=[L], **kwargs)
    
    filt = (ress[:, cols['max_ir']] == max_ir) * (ress[:, cols['L']] == L) * sp.in1d(ress[:, cols['D']], Ds)
    ress = ress[filt, :]
    
    if pars is None:
        pars = sp.unique(ress[:, cols[PARAM]])
    ccs = sp.array([sp.NaN] * len(pars))
    cc_vars = sp.array([sp.NaN] * len(pars))
    fits = [None] * len(pars)
    Dss = [None] * len(pars)
    logcls = [None] * len(pars)
    entrs = [None] * len(pars)
    
    for i in xrange(len(pars)):
        par = pars[i]
        mask = (abs(ress[:, cols[PARAM]] - par) < 1E-12)
        
        Dss[i] = ress[mask, cols['D']]
        if not sp.all(sp.in1d(Ds, Dss[i])):
            print "Not all Ds present at", par
            continue
        
        logcls[i] = sp.log2(ress[mask, cols['cl']])
        entrs[i] = ress[mask, cols['entr_max']]
        try:
            p, V = sp.polyfit(logcls[i], entrs[i], 1, cov=True)
            ccs[i] = p[0] * 6
            cc_vars[i] = V[0, 0] * 6**2
            fits[i] = sp.polyval(p, logcls[i])
        except sp.linalg.LinAlgError:
            print "cc fit error with", par
            
    nf = sp.invert(sp.isnan(ccs))
    tvar = sp.sum(ccs[nf]**2 * cc_vars[nf])
    print ccs[nf].mean(), sp.sqrt(tvar/len(ccs[nf]))
            
    plt.figure(1)       
    plt.errorbar(pars, ccs, yerr=sp.sqrt(cc_vars), fmt='bx', capsize=1.5, elinewidth=0.3, markersize=2, markeredgewidth=0.3)
    plt.xlabel('$' + PARAM_TEX + '$')
    plt.ylabel('$c$')

    plt.figure(2)
    cur_off = 0
    for i in xrange(len(pars)):        
        if not fits[i] is None:
            l, = plt.plot(logcls[i], cur_off + entrs[i], 'o', 
                          label="$" + PARAM_TEX + " = %.2f" % pars[i] + 
                          "$, $c=%.3f \\pm %.3f$" % (ccs[i], sp.sqrt(cc_vars[i])))
            plt.plot(logcls[i], cur_off + fits[i], '-', color=l.get_color())
            cur_off -= offset
    plt.xlabel("$\\log(\\xi)$")
    if cur_off == 0:
        plt.ylabel("$S$")
    else:
        plt.ylabel("$S +$ const.")
    plt.legend(loc=4)
    
    plt.show()

def calc_cf(s, op, d=20, k=0):
    ccf, ex1, ex2 = s.correlation_1s_1s(op, op, d, k=k, return_exvals=True)
        
    var = sp.zeros((s.L), dtype=sp.complex128)
    op2 = op.dot(op)
    for k in xrange(s.L):
        var[k] = s.expect_1s(op2, k) - ex1[k]**2
        
    return ccf, ex1, var

def pub_plots_prep():
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'   
    
    plt.rcParams['figure.figsize'] = [3.5, 2.5] #inches...
    plt.rcParams['figure.subplot.bottom'] = 0.2
    plt.rcParams['figure.subplot.left'] = 0.16
    plt.rcParams['figure.subplot.top'] = 0.95
    plt.rcParams['figure.subplot.right'] = 0.96
    plt.rcParams['lines.markersize'] = 2.5
    plt.rcParams['lines.markeredgewidth'] = 0
    plt.rcParams['lines.linewidth'] = 0.5
    
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 8
    
    plt.rcParams['axes.linewidth'] = 0.5

def pub_plot_gaps(usualg=False):
    import matplotlib.pyplot as plt
    
    if not usualg:
        g2i_fac = 4
        E_fac = 1
    else:
        g2i_fac = 1
        E_fac = 1.     
    
    plt.figure(figsize=(3.5, 2.5))
    plt.subplot(122)
    plt.title('(b) $O(4)$')
    #plot_gap_log(G=2, Ds=[55], max_irs=[4], lowest_en=True, usualg=True,
    #             lbl='$D=55$,\n $l \\le 2$', plot_an=True, fmt='bo')
    plot_gap_log(G=2, Ds=[91], max_irs=[4], lowest_en=True, usualg=usualg,
                 lbl='$D=91$,\n $l \\le 2$', plot_an=True, fmt='cD',
                 fcol='None', ecol='c')    
    #plot_gap_log(G=2, Ds=[91], max_irs=[3], lowest_en=True, usualg=True, 
    #             lbl='$D=91$,\n $l \\le 3/2$', plot_an=False, fmt='ro')
    plot_gap_log(G=2, Ds=[140], max_irs=[3], lowest_en=True, usualg=usualg,
                 lbl='$D=140$,\n $l \\le 3/2$', plot_an=False, fmt='r^',
                 fcol='None', ecol='r')
    plot_gap_log(G=2, Ds=[140], max_irs=[4], lowest_en=True, usualg=usualg,
                 lbl='$D=140$,\n $l \\le 2$', plot_an=False, fmt='bo',
                 fcol='None', ecol='b')
    
    plt.legend(frameon=False, loc=3, numpoints=1,
              handlelength=1, fontsize=7)
    #ax = plt.gca()
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1], frameon=False)
    
    #plt.xlim((1.5, 2.6))
    #plt.ylim((0.013, 0.28))
    plt.xlim((1.4, 0.64 * g2i_fac))
    plt.ylim((0.0075, 0.408))
    plt.ylabel('')
    #plt.gca().get_yaxis().set_ticklabels([])
    plt.gca().get_xaxis().set_ticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    plt.gca().get_xaxis().set_ticklabels(['1.4', '', '1.8', '', '2.0', '', '2.4'])
    plt.tight_layout()
    
    #plt.savefig("gap_log_SU2.pdf")

    if not usualg:
        g2i_fac = sp.sqrt(2)
        E_fac = 1. / sp.sqrt(2)
    else:
        g2i_fac = 1
        E_fac = 1
    
    #plt.figure(figsize=(3.5, 2.5))
    plt.subplot(121)
    plt.title('(a) $O(2)$')
    plot_gap_log(G=1, Ds=[32], max_irs=[6], lowest_en=True, usualg=usualg, 
                 lbl="$D=32$,\n $|n| \\le 6$", fmt='cD',
                 fcol='None', ecol='c')
    plot_gap_log(G=1, Ds=[64], max_irs=[5], lowest_en=True, usualg=usualg, 
                 lbl="$D=64$,\n $|n| \\le 5$", plot_an=False, fmt='b^',
                 fcol='None', ecol='r')
    plot_gap_log(G=1, Ds=[64], max_irs=[6], lowest_en=True, usualg=usualg, 
                 lbl="$D=64$,\n $|n| \\le 6$", plot_an=False, fmt='bo',
                 fcol='None', ecol='b')
    
    plt.legend(frameon=False, loc=3, numpoints=1,
              handlelength=1, fontsize=7)
    
    #plt.xlim((0.57, 0.93))
    #plt.ylim((0.0019, 0.73))
    plt.xlim((0.39 * g2i_fac, 0.65 * g2i_fac))
    plt.ylim((0.003 / sp.sqrt(g2i_fac) * E_fac, 1 / sp.sqrt(g2i_fac) * E_fac))
    plt.gca().get_xaxis().set_ticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9])
    plt.gca().get_xaxis().set_ticklabels(['', '0.6', '', '0.7', '', '0.8', '', '0.9'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.15, wspace=0.3)
    
    plt.savefig("gap_logs.pdf")
    
def pub_plot_beta(usualg=False):
    import matplotlib.pyplot as plt
    
    if usualg:
        g2i_fac = 4
        E_fac = 1
    else:
        g2i_fac = 1
        E_fac = 1.      
    
    plt.figure(figsize=(3.5, 2.5))
    plt.subplot(122)
    plt.title('(b) $O(4)$')
    plot_beta(G=2, Ds=[91], max_irs=[4], lowest_en=True, usualg=usualg,
                 lbl='$D=91$, \n $l \le 2$', plot_an=True, fmt='cD',
                 ecol='c', fcol='None')    
    plot_beta(G=2, Ds=[140], max_irs=[3], lowest_en=True, usualg=usualg,
                 lbl='$D=140$, \n $l \le 3/2$', plot_an=False, fmt='r^',
                 ecol='r', fcol='None')
    plot_beta(G=2, Ds=[140], max_irs=[4], lowest_en=True, usualg=usualg, 
                 lbl='$D=140$, \n $l \le 2$', plot_an=False, fmt='bo',
                 ecol='b', fcol='None')
    
    #plt.legend(frameon=False, loc=2)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc=2, numpoints=1,
              handlelength=1, fontsize=7)
    
    #plt.xlim((1.5, 2.6))
    #plt.ylim((0.013, 0.28))
    plt.xlim((0, 0.713))
    plt.ylim((0, 0.275))
    plt.ylabel('')
    plt.gca().get_xaxis().set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    plt.gca().get_xaxis().set_ticklabels(['0.0', '', '0.2', '', '0.4', '', 
                                          '0.6', ''])    
    plt.tight_layout()
    
    #plt.savefig("beta_SU2.pdf")

    if usualg:
        g2i_fac = sp.sqrt(2)
        E_fac = 1. / sp.sqrt(2)
    else:
        g2i_fac = 1
        E_fac = 1
    
    #plt.figure(figsize=(1.5, 2.5))
    plt.subplot(121)
    plt.title('(a) $O(2)$')
    plot_beta(G=1, Ds=[32], max_irs=[6], lowest_en=True, usualg=usualg,
                 lbl='$D=32$, \n $|n| \le 6$', plot_an=True, fmt='cD',
                 ecol='c', fcol='None')
    plot_beta(G=1, Ds=[64], max_irs=[5], lowest_en=True, usualg=usualg, 
                 lbl='$D=64$, \n $|n| \le 5$', plot_an=False, fmt='r^',
                 ecol='r', fcol='None')
    plot_beta(G=1, Ds=[64], max_irs=[6], lowest_en=True, usualg=usualg, 
                 lbl='$D=64$, \n $|n| \le 6$', plot_an=False, fmt='bo',
                 ecol='b', fcol='None')
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc=2, numpoints=1,
              handlelength=1, fontsize=7)
    
    #plt.xlim((0.57, 0.93))
    #plt.ylim((0.0019, 0.73))
    plt.xlim((1.0, 1.543))
    plt.ylim((0, 0.2))
    plt.tight_layout()
    
    plt.subplots_adjust(left=0.1275, right=0.99, top=0.92, bottom=0.14, wspace=0.25)
    
    plt.savefig("betas.pdf")

def pub_plot_cc():
    import matplotlib.pyplot as plt
    
    plot_cc(sp.arange(22, 88, 6), G=1, pars=[0.75, 0.8, 0.85, 0.9], 
            max_ir=5, offset=0.1)  
    plt.figure(2)
    #plt.gca().legend_.remove()
    plt.legend(loc=2, fancybox=True)
    plt.tight_layout()
    
    plt.savefig("cc_U1.pdf")
    
def pub_plot_SB_extrap():
    import matplotlib.pyplot as plt
    
    D2 = sp.array([14, 30, 55, 91, 140, 204])
    g2 = sp.array([0.502, 0.522, 0.577, 0.608, 0.653, 0.677])
    
    D1 = sp.array([32, 48, 64, 96, 128])
    g1 = sp.array([0.607, 0.613, 0.619, 0.623, 0.626])
    
    p, cov = sp.polyfit(1./D1, 1./g1, deg=1, cov=True)
    xs = sp.linspace(0, 0.05, num=2)
    
    plt.figure(figsize=[2.5, 2.])
    
    g2i_fac = 1#sp.sqrt(2)
    plt.plot(1./D1, 1./g1 / g2i_fac, '^', label='$U(1) \\sim O(2)$', markersize=3.5, markeredgewidth=0.5,
             markeredgecolor='b', markerfacecolor='None')
             
    g2i_fac = 1#4
    plt.plot(1./D2, 1./g2 / g2i_fac, 'D', label='$SU(2) \\sim O(4)$', markersize=3.5, markeredgewidth=0.5,
             markeredgecolor='r', markerfacecolor='None')
    
    g2i_fac = 1# sp.sqrt(2)     
    plt.plot(xs, sp.polyval(p, xs) / g2i_fac, 'k-', 
             label='$\\tilde g_{\mathrm{BKT}} \\approx %.3f \\pm %.3f$' % (p[1] / sp.sqrt(2),
             sp.sqrt(cov[1,1])/sp.sqrt(2)))
    #plt.gca().legend_.remove()
    l = plt.legend(loc=4, frameon=True, fancybox=True)
    l.get_frame().set_linewidth(0.4)
    plt.xlabel("$1/D$")
    #plt.ylabel("$\\tilde g_{\mathrm{SB}}$")
    plt.ylabel("$g^2_{\mathrm{SB}}$")
    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.96, top=0.96, bottom=0.17)
    
    plt.savefig("SB_extrap.pdf")

def plot_pd():
    import matplotlib.pyplot as plt
    #import matplotlib.patches as mpatches
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    
    plt.rcParams['figure.subplot.bottom'] = 0.
    plt.rcParams['figure.subplot.left'] = 0.
    plt.rcParams['figure.subplot.top'] = 1.
    plt.rcParams['figure.subplot.right'] = 1.
    
    left = -10
    right = 10
    
    t1 = 0.3
    t2 = 5
    
    ratio = 0.3
    
    paper_width = 3.4
    
    plt.figure(figsize=(paper_width, paper_width * ratio), frameon=False)
    plt.plot([left, right], [1, 1], '-', color='0.5', linewidth=1)
    plt.plot([left, t1], [1, 1], 'b--', linewidth=1)
    plt.plot([t1, t2], [1, 1], 'r--', linewidth=1)
    plt.plot([t2, right], [1, 1], 'k--', linewidth=1)
    plt.plot([left - 2, left], [1, 1], ':', color='0.5', linewidth=1)
    plt.plot([right, right + 2], [1, 1], ':', color='0.5', linewidth=1)
    
    plt.plot([0, 0], [0.5, 1.5], 'k', linewidth=0.5)
    
    plt.plot([t1, t1], [-0.5, 2.5], 'k', linewidth=1)
    plt.plot([t2, t2], [-0.5, 2.5], 'k', linewidth=1)
    
    plt.text(t1, 2.8, "second\n order", horizontalalignment="center")
    plt.text(t1, -1.3, "$\\sim %g$" % t1, horizontalalignment="center")
    plt.text(t2, 2.8, "BKT", horizontalalignment="center")
    plt.text(t2, -1.3, "$\\sim %g$" % t2, horizontalalignment="center")
    
    plt.text(left - 2, -1.3, "$\\kappa:$", horizontalalignment="left")
    
    plt.text(left, -1.3, "$%g$" % left, horizontalalignment="center")
    plt.text(right, -1.3, "$%g$" % right, horizontalalignment="center")
    
    plt.text(left + (t1 - left)/2, 1.5, "Ising", horizontalalignment="center", verticalalignment="center")
    plt.text(left + (t1 - left)/2, 0.5, "$c=0.5$", horizontalalignment="center", verticalalignment="center")
    plt.text(t1 + (t2 - t1)/2, 1.5, "floating", horizontalalignment="center", verticalalignment="center")
    plt.text(t1 + (t2 - t1)/2, 0.5, "$c=1.5$", horizontalalignment="center", verticalalignment="center")
    plt.text(t2 + (right - t2)/2, 1.5, "antiphase", horizontalalignment="center", verticalalignment="center")
    plt.text(t2 + (right - t2)/2, 0.5, "gapped", horizontalalignment="center", verticalalignment="center")
    
    ax = plt.gca()
    #p = mpatches.Circle((1.1, 1), 0.2, fc="w")
    #ax.add_patch(p)
    #ax.annotate("$c=1.5$", (1.1, 1), (1.1, -2.5), ha="center", arrowprops=dict(arrowstyle='->',))
    #ax.annotate("$c=0.5$", (-10., 1), (-10., -2.5), ha="center", arrowprops=dict(arrowstyle='->',))
    #ax.annotate("$c=0.5$", (-2., 1), (-2., -2.5), ha="center", arrowprops=dict(arrowstyle='->',))
    #ax.annotate("$c=0.5$", (-1., 1), (-1., -1), ha="center", arrowprops=dict(arrowstyle='->',))
    #ax.annotate("no CFT", (10, 1), (10, -2.5), ha="center", arrowprops=dict(arrowstyle='->',))
    
    height = (right - left + 4) * ratio
    plt.ylim(1 - height / 2, 1 + height / 2)
    plt.xlim(left - 2, right + 2)
    
    ax.set_axis_off()
#    ax.get_yaxis().set_visible(False)
#    ax.spines['left'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
    #plt.tight_layout()
    plt.show()

def get_ground(s, D_targ, ops, step_init=0.04, tol=1E-8, expand_tol=None, D_step=2, 
               use_CG=True, par_str='', CG_start_tol=1E-1):
    print "Bond dimension: " + str(s.D)
    print
    col_heads = ["par", "Step", "<h>", "d<h>", "corrlen"] + ["Sz"] * s.L + ["eta"]
    print "\t".join(col_heads)
    print
    
    U, P, P2, ReUUH, ReUUH_tp = ops

    """
    Create a function to print out info during the solver iterations.
    """
    h_prev = [0]
    def cbf(s, i, **kwargs):
        h = s.h_expect.real

        row = []
        
        row.append(par_str)
        
        row.append(str(i))

        row.append("%.15g" % h)

        dh = h - h_prev[0]
        h_prev[0] = h

        row.append("%.2e" % (dh))
        
        if i % 10 == 0:
            cl = s.correlation_length().real
        else:
            cl = sp.nan
        row.append("%.2e" % (cl))

        """
        Compute expectation values!
        """
        exUs = []
        for k in xrange(s.L):
            exUs.append("%.3g" % s.expect_1s(U[0][0], k=k).real)
        row += exUs
        
        exP2s = []
        for k in xrange(s.L):
            exP2s.append("%.3g" % s.expect_1s(P2, k=k).real)
        row += exP2s
        
        row += map(lambda k: "%.6g" % s.entropy(k).real, range(s.L))
        
        row.append("%.6g" % s.eta.real)

        row.append(str(kwargs))

        print "\t".join(row)


    if use_CG:
        dy.find_ground(s, tol=tol, h_init=step_init, expand_to_D=D_targ, 
                       expand_step=D_step, expand_tol=expand_tol, cb_func=cbf,
                       CG_start_tol=CG_start_tol)
    else:
        dy.opt_im_time(s, tol=tol, dtau_base=step_init, expand_to_D=D_targ,
                       cb_func=cbf, auto_trunc=True)
                       
    if s.eta.real < tol:
        s.update()
        return s
                       

def calc_excite(s, num_mom=20, num_exc=20, v0=None):
    """
    Find excitations if we have the ground state.
    """
    print 'Finding excitations!'

    ex_ev = []
    ex_p = []
    for p in sp.linspace(0, sp.pi, num=num_mom):
        print "p = ", p
        ev, eV = s.excite_top_triv(p, k=num_exc, ncv=num_exc * 4,
                                   return_eigenvectors=True, v0=v0)
        print ev
        
        #Use previous momentum lowest eigenvector as a starting point
        ind = ev.argmin()
        v0 = eV[:, ind]
        
        ex_ev.append(ev)        
        ex_p.append([p] * num_exc)
    
    ex_ev = sp.array(ex_ev).ravel()
    ex_p = sp.array(ex_p).ravel()
    return sp.column_stack((ex_p, ex_ev)), v0
    
def get_grid_1d(a, b, step=0.0001, dp=4):
    a = sp.around(a * 10**dp)
    b = sp.around(b * 10**dp)
    step = sp.around(step  * 10**dp)
    
    return sp.arange(a, b + step, step, dtype=int) / 10.0**dp
    
def load_state(G, g2inv, D, L, max_ir, loc='state_data/', par_tol=1E-8, 
               from_file=None, ops=None, ret_res_row=False):
    if from_file is None:
        ress = sp.genfromtxt(RES_LOAD_FROM, delimiter="\t",
                             dtype=None)
        intref = sp.array([G, D, L, max_ir], dtype=int)
        fns = []
        ens = []
        resrs = []
        for res in ress:
            ints = sp.array([res[cols[s]] for s in ['G', 'D', 'L', 'max_ir']], dtype=int)
            if (sp.all(ints == intref) and 
              sp.allclose(res[cols['g2inv']], g2inv, atol=par_tol)):
                fns.append(res[cols['fn']])
                ens.append(res[cols['energy']])
                resrs.append(res)
    else:
        fns = [from_file]
        ens = [0]
        resrs = [None]
    
    print ens
    print fns
    if len(fns) > 0:
        ens = sp.array(ens)
        ind = ens.argmin()
        fn = fns[ind]
        print "Choosing lower energy, delta =", ens.max() - ens[ind]
        resr = resrs[ind]
        
        if ops is None:
            ops = get_ops(G, max_ir)
        s = create_tdvp(D, get_ham(g2inv, ops), L=L, ham_tp=get_ham_tp(g2inv, ops))
        s.load_state(loc + fn)
        if ret_res_row:
            if resr is None:
                resr = get_res_row(s, G, g2inv, max_ir, ops=ops)
            return s, resr
        else:
            return s
    else:
        print "No state found!", g2inv, D, L
        if ret_res_row:
            return None, None
        else:
            return None

def calc_wv_gap(s_in, v0=None, blocking_D_limit=24, calc_gap=True, nev=20):    
    evs = s_in._calc_E_largest_eigenvalues(k=4, ncv=None)
    if len(evs) > 1:
        ind = abs(evs).argmax()
        mask = sp.ones((len(evs)), dtype=bool)
        mask[ind] = False
        ind_wv = abs(evs[mask]).argmax()
        wv = sp.angle(evs[mask][ind_wv])
    else:
        wv = 0.
    
    if calc_gap:
        if s_in.L > 1 and s_in.D <= blocking_D_limit:
            s = copy.deepcopy(s_in)
            s.convert_to_TI_blocked()
        else:
            s = s_in

        try:
            evs, eVs = s.excite_top_triv(wv, nev=4, v0=v0, return_eigenvectors=True, tol=1E-11, pinv_tol=1E-12)
            ind = evs.argmin()
            v0 = eVs[:, ind]
            gap = evs[ind]
        except tdvp.EvoMPSNoConvergence:
            gap = sp.NaN
    else:
        gap = sp.NaN
        
    return wv, gap, v0
        
def scan(g2inv1, g2inv2, G=1, step=0.01, D=16, D_init=8, D_step=2, L=1, max_ir=5,
            tol=1E-8, expand_tol=None, CG_start_tol=1E-1, h_init=0.04,
            load_first=True, load_tol=1E-8, start_state=None, calc_gap=True):
    pars = get_grid_1d(g2inv1, g2inv2, step)
    
    ops = get_ops(G, max_ir)
    
    if not start_state is None:
        s = start_state
    elif load_first:
        s = load_state(G, pars[0], D, L, max_ir, par_tol=load_tol, ops=ops)
        if not s is None:
            if load_tol <= 1E-8:
                pars = pars[1:]
        else:
            s = load_state(G, pars[0], D_init, L, max_ir, par_tol=load_tol, ops=ops)
            print "Trying to load at D_init for", pars[0]
            
        if s is None:
            print "State not found for", pars[0]
            return
    else:
        s = None
        
    v0 = None
    for par in pars:
        if s is None:
            s = create_tdvp(D_init, get_ham(par, ops), L=L, ham_tp=get_ham_tp(par, ops))
        else:
            s.set_ham(get_ham(par, ops))
            s.ham_tp = get_ham_tp(par, ops)
            
        s = get_ground(s, D, ops, tol=tol, par_str=str([par, D, L]), 
                       D_step=D_step, expand_tol=expand_tol, 
                       CG_start_tol=CG_start_tol, step_init=h_init)
        if not s is None:
            wv, gap, v0 = calc_wv_gap(s, v0=v0, calc_gap=calc_gap)
            resrow = save_result(s, G, par, max_ir, wv=wv, gap_wv=gap, v0=v0, ops=ops)
        else:
            resrow = None
            print "Failed to get ground for", par
            
    return s, resrow
          
def fg_one(g2inv, D, G=1, L=1, max_ir=5, tol=1E-8, load_tol=1E-1, calc_gap=True):
    scan(g2inv, g2inv, G=G, D=D, D_init=D, L=L, max_ir=max_ir, tol=tol, load_first=True,
            load_tol=load_tol, calc_gap=calc_gap)

def fine_grain(g2inv1, g2inv2, G=1, step=0.01, D=16, L=1, max_ir=5, tol=1E-8, 
               mprocs=1, load_tol=1E-1, calc_gap=True):
    pars = get_grid_1d(g2inv1, g2inv2, step)
    
    if mprocs > 1:
        from multiprocessing import Pool
        p = Pool(mprocs)
        resobjs = [p.apply_async(fg_one, args=(par, D), 
                                 kwds={'L': L,
                                       'max_ir': max_ir,
                                       'tol': tol,
                                       'load_tol': load_tol,
                                       'calc_gap': calc_gap,
                                       'G': G})
                   for par in pars]
        for ro in resobjs:
            ro.get()
    else:
        for par in pars:
            fg_one(par, D, G=G, L=L, max_ir=max_ir, tol=tol, load_tol=load_tol,
                   calc_gap=calc_gap)
          
def expand_one(g2inv, D_init, D, D_step, G=1, L=1, max_ir=5, tol=1E-8, calc_gap=True, 
               load_from=None, D_step_intern=2, expand_tol=None,
               until_conv_in=None, conv_tol=1E-4, try_to_load_allD=False):
    D_targ = D_init
    s, resr = load_state(G, g2inv, D_init, L, max_ir, from_file=load_from,
                         ret_res_row=True)
    
    conv_vals = []
    sL = None
    while D_targ < D:
        D_prev = D_targ
        D_targ += D_step
        if not until_conv_in is None:
            conv_vals.append(resr[cols[until_conv_in]])
            if len(conv_vals) > 1:
                diff = abs((conv_vals[-1] - conv_vals[-2]) / conv_vals[-1])
                print "Convergence check (" + until_conv_in + "):", conv_vals, diff
                if diff < conv_tol:
                    print "Converged!"
                    break
        if try_to_load_allD:
            sL, resrL = load_state(G, g2inv, D_targ, L, max_ir, ret_res_row=True)
        if sL is None:
            s, resr = scan(g2inv, g2inv, G=G, D=D_targ, D_init=D_prev, L=L, 
                           max_ir=max_ir, tol=tol, 
                           load_first=False, start_state=s, calc_gap=calc_gap,
                           D_step=D_step_intern, expand_tol=expand_tol)
        else:
            s = sL
            resr = resrL

def expand_existing(g2inv1, g2inv2, G=1, step=0.01, D=16, D_init=8, D_step=2, L=1, 
                    max_ir=5, until_conv_in=None, conv_tol=1E-4,
                    tol=1E-8, mprocs=1, calc_gap=True, D_step_intern=2,
                    expand_tol=None, try_to_load_allD=False):
    pars = get_grid_1d(g2inv1, g2inv2, step)
    
    if mprocs > 1:
        from multiprocessing import Pool
        p = Pool(mprocs)
        resobjs = [p.apply_async(expand_one, args=(par, D_init, D, D_step), 
                                 kwds={'G': G,
                                       'L': L,
                                       'max_ir': max_ir,
                                       'tol': tol,
                                       'calc_gap': calc_gap,
                                       'D_step_intern': D_step_intern,
                                       'expand_tol': expand_tol,
                                       'until_conv_in': until_conv_in,
                                       'conv_tol': conv_tol,
                                       'try_to_load_allD': try_to_load_allD})
                   for par in pars]
        for ro in resobjs:
            ro.get()
    else:
        for par in pars:
            expand_one(par, D_init, D, D_step, G=G, L=L, max_ir=max_ir, tol=tol, 
                       calc_gap=calc_gap, D_step_intern=D_step_intern,
                       expand_tol=expand_tol, until_conv_in=until_conv_in,
                       conv_tol=conv_tol, try_to_load_allD=try_to_load_allD)

#def gap_existing_one(g2inv, D, G=1, L=1, max_ir=5, tol=1E-8, calc_gap=True, 
#                     load_from=None, excite_tol=0, pinv_tol=1E-12):
#    s, resr = load_state(G, g2inv, D, L, max_ir, from_file=load_from,
#                         ret_res_row=True)
#    
#    
#                       
#def gap_existing(g2inv1, g2inv2, G=1, step=0.01, D=16, D_init=8, D_step=2, L=1, 
#                    max_ir=5, until_conv_in=None, conv_tol=1E-4,
#                    tol=1E-8, mprocs=1, calc_gap=True, D_step_intern=2,
#                    expand_tol=None, try_to_load_allD=False):
#    pars = get_grid_1d(g2inv1, g2inv2, step)
#    
#    if mprocs > 1:
#        from multiprocessing import Pool
#        p = Pool(mprocs)
#        resobjs = [p.apply_async(gap_existing_one, args=(par, D_init, D, D_step), 
#                                 kwds={'G': G,
#                                       'L': L,
#                                       'max_ir': max_ir,
#                                       'tol': tol,
#                                       'calc_gap': calc_gap,
#                                       'D_step_intern': D_step_intern,
#                                       'expand_tol': expand_tol,
#                                       'until_conv_in': until_conv_in,
#                                       'conv_tol': conv_tol,
#                                       'try_to_load_allD': try_to_load_allD})
#                   for par in pars]
#        for ro in resobjs:
#            ro.get()
#    else:
#        for par in pars:
#            gap_existing_one(par, D_init, D, D_step, G=G, L=L, max_ir=max_ir, tol=tol, 
#                       calc_gap=calc_gap, D_step_intern=D_step_intern,
#                       expand_tol=expand_tol, until_conv_in=until_conv_in,
#                       conv_tol=conv_tol, try_to_load_allD=try_to_load_allD)
        
def reprocess_existing(new_res_file, recalc_gap=False):
    oldcols = {'G':        0,
            'g2inv':    1, 
            'D':        2, 
            'L':        3,
            'max_ir':   4,
            'eta':      5, 
            'energy':   6, 
            'cl':       7, 
            'entr_max': 8, 
            'U_av':     9, 
            'U_0':      10,
            'U2_0':     11,
            'P_av':     12,
            'P_0':      13,
            'P2_0':     14,
            'ReUUH_av': 15,
            'ReUUH_0':  16,
            'ReUUH2_0': 17,
            'wv_dom':   18, 
            'wv_fit':   19, 
            'gap':      20, 
            'exc_fn':   21,
            'fn':       22}    
    
    ress = sp.genfromtxt(RES_LOAD_FROM, delimiter="\t", dtype=None)
                         
    for res in ress:
        print "processing:", res
        G = res[oldcols['G']]
        g2inv = res[oldcols['g2inv']]
        D = res[oldcols['D']]
        L = res[oldcols['L']]
        max_ir = res[oldcols['max_ir']]
        fn = res[oldcols['fn']]
        
        wv = res[oldcols['wv_dom']]
        gap = res[oldcols['gap']]
        
        ops = get_ops(G, max_ir)
        s = create_tdvp(D, get_ham(g2inv, ops), L=L, ham_tp=get_ham_tp(g2inv, ops))
        s.load_state('state_data/' + fn)
        s.update()
        s.calc_B()
        if recalc_gap:# or sp.isnan(gap):
            wv, gap, v0 = calc_wv_gap(s)

        save_result(s, G, g2inv, max_ir, wv=wv, gap_wv=gap, existing_fn=fn, 
                    res_file=new_res_file, ops=ops)
                    
def calc_gap_existing(new_res_file, g2inv1, g2inv2, G=1, step=0.01, D=16,  
                      L=1, max_ir=5, recalc=False, max_U00=1000):
    pars = get_grid_1d(g2inv1, g2inv2, step)
    
    ops = get_ops(G, max_ir)                          
                          
    ress = sp.genfromtxt(RES_LOAD_FROM, delimiter="\t", dtype=None)

    resf = open(new_res_file, 'a')
    fcntl.flock(resf, fcntl.LOCK_EX)
    
    v0 = None
                         
    for res in ress:
        if (G == res[cols['G']]
        and D == res[cols['D']]
        and L == res[cols['L']]
        and max_ir == res[cols['max_ir']]
        and abs(res[cols['U_0']]) < max_U00
        and sp.any(abs(pars - res[cols['g2inv']]) < 1E-8)
        and (recalc or sp.isnan(res[cols['gap']]))):
            print "processing:", res
            g2inv = res[cols['g2inv']]
            fn = res[cols['fn']]
            s = create_tdvp(D, get_ham(g2inv, ops), L=L, ham_tp=get_ham_tp(g2inv, ops))
            s.load_state('state_data/' + fn)
            s.update()
            s.calc_B()
            wv, gap, v0 = calc_wv_gap(s, v0=v0)
            print "Gap:", gap, "previously", res[cols['gap']]
            res[cols['wv_dom']] = wv
            res[cols['gap']] = gap
            resf.flush()
    
        resf.write("\t".join(map(str, res)) + "\n")
        
    resf.close()
    #sp.savetxt(new_res_file, ress)
        #save_result(s, G, g2inv, max_ir, wv=wv, gap_wv=gap, existing_fn=fn, 
        #            res_file=new_res_file, ops=ops)
        
def move_data():
    ress = sp.genfromtxt('petal_res.txt', delimiter="\t", usecols=(-1), dtype=None)
    for res in ress:
        shutil.move(res, 'state_data/' + res)
