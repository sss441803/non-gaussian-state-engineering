import numpy as np
import torch




def sympmat(N):
    I = torch.eye(N, dtype=complex_type, device=device)
    O = torch.zeros_like(I, dtype=complex_type, device=device)
    S = torch.cat((torch.cat((O, I), dim=1), torch.cat((-I, O), dim=1)), dim=0)
    return S

def williamson(V, tol=1e-11):
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    diffn = torch.norm(V - V.t())

    if diffn >= tol:
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = sympmat(n)
    vals, _ = torch.symeig(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = torch.linalg.inv(V).real.sqrt()
    r1 = Mm12 @ omega @ Mm12
    s1, K = torch.schur(r1)
    X = torch.tensor([[0, 1], [1, 0]], device=device)
    I = np.identity(2)
    seq = []

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = xpxp_to_xxpp(s1t)
    perm_indices = xpxp_to_xxpp(np.arange(2 * n))
    Ktt = Kt[:, perm_indices]
    Db = np.diag([1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T


def thermal_photons(nth, cutoff = 20):
    return 1 / (nth + 1) * (nth / (nth + 1)) ** torch.arange(cutoff)

def get_cumsum_kron(sq_cov, L, chi = 100, max_dim = 10 ** 5, cutoff = 6, err_tol = 10 ** (-12)):
    M = len(sq_cov) // 2
    mode = np.arange(L, M)
    modes = np.append(mode, mode + M)
    sq_cov_A = sq_cov[np.ix_(modes, modes)]

    D, S = williamson(sq_cov_A)
    d = (np.diag(D) - 1) / 2

    d[d < 0] = 0

    res = thermal_photons(d[0], cutoff)
    num = np.arange(cutoff, dtype='int8')
    
    kron_time = 0
    cart_time = 0
    select_time = 0
    sort_time = 0
    rev_time = 0
    
    for i in range(1, M - L):
        start = time.time()
        res = np.kron(res, thermal_photons(d[i], cutoff))
        kron_time += time.time() - start
        start = time.time()
        keep_idx = np.where(res > err_tol)[0]
        start = time.time()
        if len(num.shape) == 1:
            num = num.reshape(-1, 1)
        num = np.concatenate([num[keep_idx // cutoff], np.arange(cutoff).reshape(-1, 1)[keep_idx % cutoff]], axis=1)
        cart_time += time.time() - start
        res = res[keep_idx]
        select_time += time.time() - start
        start = time.time()
        idx = np.argsort(res)[-min(len(res), max_dim):]       
        sort_time += time.time() - start
        start = time.time()
        res = res[idx][::-1]
        num = num[idx][::-1]
        rev_time += time.time() - start

    # print('loop time ', kron_time, cart_time, select_time, sort_time, rev_time)
            
    len_ = min(chi, len(res))
    idx = np.argsort(res)[-len_:]
    idx_sorted = idx[np.argsort(res[idx])]
    res = res[idx_sorted][::-1]
    num = num[idx_sorted][::-1]
    # print(res.shape, num.shape)

    return res, num, S