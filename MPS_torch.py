import numpy as np
import torch
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--dir', type=str, help="Root directory.", default=0)
args = vars(parser.parse_args())

d = args['d']
chi = args['chi']
rootdir = args['dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
complex_type = torch.complex64

def Sigma_select(Sigma, target):
    batch_size = 65535
    n_batch, n_select = target.shape
    Sigma2 = torch.zeros([n_batch, n_select, n_select], dtype=complex_type, device=device)
    for batch_id in range(n_batch // batch_size + 1):
        begin = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_batch)
        batch_target = target[begin : end]
        rows = np.repeat(batch_target, n_select, axis=1).reshape(-1)
        cols = np.tile(batch_target, (1, n_select)).reshape(-1)
        Sigma2[begin : end] = Sigma[rows, cols].reshape(end - begin, n_select, n_select)
    return Sigma2
    
def push_to_end(array):
    n_batch, n_select = array.shape
    new_array = torch.zeros_like(array, device=device)
    idx = np.zeros(n_batch, dtype='int32')
    for i in range(1, n_select + 1):
        occupied = array[:, -i] != 0
        idx += occupied
        new_array[np.arange(n_batch), -idx] = array[:, -i]
    return new_array

def xpxp_to_xxpp(S):
    shape = S.shape
    n = shape[0]

    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(-1, 2).T.flatten()

    if len(shape) == 2:
        if shape[0] != shape[1]:
            raise ValueError("The input matrix is not square")
        return S[:, ind][ind]

    return S[ind]

def sqrtm(matrix):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def hafnian(A):
    matshape = A.shape[1:]
    n_batch = A.shape[0]
    if matshape == (0, 0):
        return torch.ones(n_batch, dtype=complex_type)

    if matshape[0] % 2 != 0:
        return torch.zeros(n_batch, dtype=complex_type)
    '''removed case where it is identity'''
    if matshape[0] == 2:
        return A[:, 0, 1]
    if matshape[0] == 3:
        return (
            A[:, 0, 0] * A[:, 1, 2] + A[:, 1, 1] * A[:, 0, 2] + A[:, 2, 2] * A[:, 0, 1] + A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
        )
    if matshape[0] == 4:
        return A[:, 0, 1] * A[:, 2, 3] + A[:, 0, 2] * A[:, 1, 3] + A[:, 0, 3] * A[:, 1, 2]
    return recursive_hafnian(A)

def recursive_hafnian(A):  # pragma: no cover
    n_batch, nb_lines, nb_columns = A.shape
    if nb_lines != nb_columns:
        raise ValueError("Matrix must be square")
    if nb_lines % 2 != 0:
        raise ValueError("Matrix size must be even")
    n = A.shape[1] // 2
    z = torch.zeros((n_batch, n * (2 * n - 1), n + 1), dtype=A.dtype, device=A.device)
    for j in range(1, 2 * n):
        ind = j * (j - 1) // 2
        for k in range(j):
            z[:, ind + k, 0] = A[:, j, k]
    g = torch.zeros([n_batch, n + 1], dtype=A.dtype, device=A.device)
    g[:, 0] = 1
    return solve(z, 2 * n, 1, g, n)

def solve(b, s, w, g, n):  # pragma: no cover
    n_batch = b.shape[0]
    if s == 0:
        return w * g[:, n]
    c = torch.zeros((n_batch, (s - 2) * (s - 3) // 2, n + 1), dtype=b.dtype, device=b.device)
    i = 0
    for j in range(1, s - 2):
        for k in range(j):
            c[:, i] = b[:, (j + 1) * (j + 2) // 2 + k + 2]
            i += 1
    h = solve(c, s - 2, -w, g, n)
    e = g.clone()
    for u in range(n):
        for v in range(n - u):
            e[:, u + v + 1] += g[:, u] * b[:, 0, v]
    for j in range(1, s - 2):
        for k in range(j):
            for u in range(n):
                for v in range(n - u):
                    c[:, j * (j - 1) // 2 + k, u + v + 1] += (
                        b[:, (j + 1) * (j + 2) // 2, u] * b[:, (k + 1) * (k + 2) // 2 + 1, v]
                        + b[:, (k + 1) * (k + 2) // 2, u] * b[:, (j + 1) * (j + 2) // 2 + 1, v]
                    )
    return h + solve(c, s - 2, w, e, n)

def blochmessiah(S):
    N, _ = S.shape
    device = S.device
    # Changing Basis
    eye = torch.eye(N // 2, device=device)
    R = torch.cat([torch.cat([eye,  1j * eye], dim=1), 
                   torch.cat([eye, -1j * eye], dim=1)], dim=0) / np.sqrt(2)
    Sc = R @ S @ R.conj().T
    # Polar Decomposition
    # u1, d1, v1 = np.linalg.svd(Sc)
    u1, d1, v1 = torch.linalg.svd(Sc, lapack_driver='gesvd')
    Sig = u1 @ torch.diag(d1) @ np.conjugate(u1).T
    Unitary = u1 @ v1
    # Blocks of Unitary and Hermitian symplectics
    alpha = Unitary[0 : N // 2, 0 : N // 2]
    beta = Sig[0 : N // 2, N // 2 : N]
    # Bloch-Messiah in this Basis
    u2, d2, v2 = torch.linalg.svd(beta)
    sval = np.arcsinh(d2)
    takagibeta = u2 @ sqrtm(np.conjugate(u2).T @ (v2.T))
    uf = torch.block_diag(takagibeta, takagibeta.conj())
    vf = torch.block_diag(takagibeta.conj().T @ alpha, (takagibeta.conj().T @ alpha).conj())
    df = torch.cat([torch.cat([torch.diag(torch.cosh(sval)), torch.diag(torch.sinh(sval))]),
                    torch.cat([torch.diag(torch.sinh(sval)), torch.diag(torch.cosh(sval))])], dim=0)
    # Rotating Back to Original Basis
    uff = R.conj().T @ uf @ R
    vff = R.conj().T @ vf @ R
    dff = R.conj().T @ df @ R
    return uff, dff, vff

def get_Sigma(U2, sq, U1):
    device = U2.device
    M = len(sq)
    Sigma = torch.zeros((2 * M, 2 * M), dtype=complex_type, device=device)
    Sigma[:M, :M] = U2 @ torch.diag(torch.tanh(sq)) @ U2.T
    Sigma[:M, M:] = U2 @ torch.diag(1 / torch.cosh(sq)) @ U1
    Sigma[M:, :M] = U1.T @ torch.diag(1 / torch.cosh(sq)) @ U2.T
    Sigma[M:, M:] = -U1.T @ torch.diag(torch.tanh(sq)) @ U1
    return Sigma

def get_target(num):
    n_select = np.sum(num, axis=1).max()
    n_batch, n_len = num.shape
    idx_x = np.tile(np.arange(n_select, dtype='int32').reshape(1, -1), (n_batch, 1))
    target = np.zeros([n_batch, n_select], dtype='int32')
    idx_end = np.zeros(n_batch)
    idx_begin = np.zeros(n_batch)
    num = np.array(num)
    for i in range(n_len):
        vals_n = num[:, i]
        idx_end += vals_n
        mask = idx_x >= idx_begin.reshape(-1, 1)
        mask *= idx_x < idx_end.reshape(-1, 1)
        target += mask * (i + 1)
        idx_begin = np.copy(idx_end)
    return torch.tensor(target)

def A_elem(Sigma, target, denominator, max_memory_in_gb):
    # print(target.shape)
    n_batch, n_select = target.shape
    all_haf = torch.zeros([0], dtype=complex_type, device=Sigma.device)
    if n_select == 0:
        n_batch_max = 99999999999
    else:
        n_batch_max = int(max_memory_in_gb * (10 ** 9) // (n_select ** 2 * 8))
    # print(n_batch_max)
    sigma_time = 0
    haf_time = 0
    for begin_batch in range(0, n_batch, n_batch_max):
        end_batch = min(n_batch, begin_batch + n_batch_max)
        start = time.time()
        Sigma2 = Sigma_select(Sigma, target[begin_batch : end_batch])
        sigma_time += time.time() - start
        start = time.time()
        haf = hafnian(Sigma2)
        haf_time += time.time() - start
        # print(haf)
        all_haf = torch.cat([all_haf, haf], dim=0)
    return all_haf / denominator, haf_time, sigma_time

def get_U2_sq_U1(S_l, S_r):
    M = len(S_r) // 2
    mode = np.arange(M - 1) + 1
    modes = np.append(mode, mode + M)
    S_l2_inv = torch.eye(2 * M, dtype = torch.float)
    S_l2_inv[np.ix_(modes, modes)] = torch.linalg.inv(S_l)
    S = S_l2_inv @ S_r
    S2, SQ, S1 = blochmessiah(S)
    U2 = S2[:M, :M] - 1j * S2[:M, M:]
    U1 = S1[:M, :M] - 1j * S1[:M, M:]
    sq = torch.log(torch.diag(SQ)[:M])
    return U2, sq, U1




if __name__ == "__main__":

    path = rootdir + f"d_{d}_chi_{chi}/"
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    M = len(cov) // 2

    for compute_site in range(M):
        print('mode: ', compute_site)

        real_start = time.time()

        max_memory_in_gb = 0.5
        max_dim = 10 ** 5; err_tol = 10 ** (-10)
        tot_a_elem_time = 0
        tot_haf_time = 0
        tot_sigma_time = 0

        _, S_r = williamson(sq_cov)

        Gamma = np.zeros([chi, chi, d], dtype='complex64')
        Lambda = np.zeros([chi], dtype='float32')



        if compute_site == 0:

            res = np.load(path + f'res_{compute_site}.npy')
            num = np.load(path + f'num_{compute_site}.npy')
            S_l = np.load(path + f'S_{compute_site}.npy')
            num = num[res > err_tol]
            res = res[res > err_tol]
            U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
            Sigma = get_Sigma(U2, sq, U1)
            left_target = get_target(num)
            left_sum = np.sum(num, axis=1)
            left_denominator = np.sqrt(np.product(np.array(factorial(num)), axis=1))
            Z = np.sqrt(np.prod(np.cosh(sq)))
            Lambda[:len(res)] = np.array(np.sqrt(res))
            for j in np.arange(d):
                print('j: ', j)
                for size in np.arange(np.max(left_sum) + 1):
                    left_idx = np.where(left_sum == size)[0]
                    if (Lambda[left_idx] <= err_tol).all():
                        continue
                    n_batch = left_idx.shape[0]
                    '''one is already added to the left charge in function get_target'''
                    target = np.append(np.zeros([n_batch, j], dtype='int32'), left_target[:, :size][left_idx], axis=1)
                    denominator = np.sqrt(factorial(j)) * left_denominator[left_idx]
                    haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                    tot_haf_time += haf_time
                    Gamma[0, left_idx, j] = haf / Z / Lambda[left_idx]

        elif compute_site == M - 1:

            num_pre = np.load(path + f'num_{compute_site - 1}.npy')
            num_pre = num_pre.reshape(num_pre.shape[0], -1)
            S_r = np.load(path + f'S_{compute_site - 1}.npy')
            right_target = get_target(num_pre)
            right_sum = np.array(np.sum(num_pre, axis=1))
            right_denominator = np.sqrt(np.product(np.array(factorial(num_pre)), axis=1))

            S_l = np.zeros((0, 0))
            U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
            Z = np.sqrt(np.prod(np.cosh(sq)))
            Sigma = get_Sigma(U2, sq, U1)

            for j in np.arange(d):
                print('j: ', j)
                for size in np.arange(int(np.nanmax(right_sum)) + 1):
                    right_idx = np.where(right_sum == size)[0]
                    n_batch = right_idx.shape[0]
                    if size == 0 and j == 0:
                        Gamma[right_idx, 0, j] = np.ones(n_batch) / Z
                        continue

                    target = np.copy(right_target[:, :size][right_idx])
                    if size == 0:
                        target = np.zeros([n_batch, 0], dtype='int32')
                    target = np.append(np.zeros([n_batch, j], dtype=int), target, axis=1)
                    denominator = np.sqrt(factorial(j)) * right_denominator[right_idx]
                    haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                    Gamma[right_idx, 0, j] = haf / Z

        else:
                    
            num_pre = np.load(path + f'num_{compute_site - 1}.npy')
            res_pre = np.load(path + f'res_{compute_site - 1}.npy')
            S_r = np.load(path + f'S_{compute_site - 1}.npy')
            right_target = np.array(push_to_end(get_target(num_pre)))
            right_sum = np.array(np.sum(num_pre, axis=1))
            right_denominator = np.sqrt(np.product(np.array(factorial(num_pre)), axis=1))

            num = np.load(path + f'num_{compute_site}.npy')
            res = np.load(path + f'res_{compute_site}.npy')
            S_l = np.load(path + f'S_{compute_site}.npy')
            num = num[res > err_tol]
            num = num.reshape(num.shape[0], -1)
            left_target = get_target(num)
            left_n_select = left_target.shape[1]
            left_sum = np.array(np.sum(num, axis=1))
            full_sum = np.repeat(left_sum.reshape(-1, 1), right_sum.shape[0], axis=1) + np.repeat(right_sum.reshape(1, -1), left_sum.shape[0], axis=0)
            left_denominator = np.sqrt(np.product(np.array(factorial(num)), axis=1))
            res = res[res > err_tol]
            U2, sq, U1 = get_U2_sq_U1(S_l, S_r) # S_l: left in equation, S_r : right in equation
            Sigma = get_Sigma(U2, sq, U1)
            Z = np.sqrt(np.prod(np.cosh(sq)))
            Lambda[:len(res)] = np.array(np.sqrt(res))

            for j in np.arange(d):
                print('j: ', j)
                gpu_Gamma = np.zeros([chi, chi], dtype='complex64')
                for size in np.arange(int(np.nanmax(full_sum)) + 1):
                    left_idx, right_idx = np.where(full_sum == size)
                    n_batch = left_idx.shape[0]
                    if (Lambda[left_idx] <= err_tol).all():
                        continue
                    if size == 0 and j == 0:
                        gpu_Gamma[right_idx, left_idx] = np.ones(n_batch) / Z / Lambda[left_idx]
                        continue
                    if size == 0:
                        n_batch_max = 99999999999
                    else:
                        n_batch_max = int(max_memory_in_gb * (10 ** 9) // (size * 8))
                    for begin_batch in range(0, n_batch, n_batch_max):
                        end_batch = min(n_batch, begin_batch + n_batch_max)
                        target = np.zeros([end_batch - begin_batch, size], dtype='int32')
                        target[:, :left_n_select] = np.copy(left_target[:, :size][left_idx[begin_batch : end_batch]])
                        right_target_chosen = np.copy(right_target[:, -size:][right_idx[begin_batch : end_batch]])
                        if size == 0:
                            right_target_chosen = np.zeros([end_batch - begin_batch, 0], dtype='int32')
                        right_n_select = right_target_chosen.shape[1]
                        non_zero_locations = np.where(right_target_chosen != 0)
                        right_target_chosen[non_zero_locations] += num.shape[1]
                        target[:, -right_n_select:] += right_target_chosen
                        target = np.append(np.zeros([end_batch - begin_batch, j], dtype=int), target, axis=1)
                        denominator = np.sqrt(factorial(j)) * left_denominator[left_idx[begin_batch : end_batch]] * right_denominator[right_idx[begin_batch : end_batch]]
                        start = time.time()
                        haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                        tot_a_elem_time += time.time() - start
                        tot_haf_time += haf_time
                        tot_sigma_time += sigma_time
                        gpu_Gamma[right_idx[begin_batch : end_batch], left_idx[begin_batch : end_batch]] = haf / Z / Lambda[left_idx[begin_batch : end_batch]]
                Gamma[:, :, j] = gpu_Gamma
        print('Total {}, a_elem {}, haf {}, sigma {}.'.format(time.time() - real_start, tot_a_elem_time, tot_haf_time, tot_sigma_time))

        np.save(path + f"Gamma_{compute_site}.npy", Gamma)
        if compute_site < M - 1:
            np.save(path + f"Lambda_{compute_site}.npy", Lambda)