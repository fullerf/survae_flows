import torch


def mode_k_prod(A, B, k=0):
    """
    A generalized implementation for any size tensor A(i1,i2,...,ik,...,iN)
    and a matrix B(alpha,ik) such that:

    we return the mode_k product C(i1,i2,...,alpha,...,iN) = sum_{ik}{A*B}
    """
    return torch.moveaxis(torch.tensordot(A, B, [[int(k)], [-1]]), -1, int(k))


def horizontal_unfold(A):
    """
    For a 3D tensor A(a,i,b), we unfold like: A(a,ib)
    """
    S = A.shape
    return A.reshape(S[0], S[1] * S[2])


def horizontal_refold(Au, i: int):
    """
    Assuming that A was a 3D tensor that was horizontally unfolded to Au, this
    returns the original 3D tensor A. Must suply the unfolded tensor Au and
    the length of the central dimension i
    """
    S = Au.shape
    r3 = S[1] // i
    return Au.reshape(S[0], i, r3)


def vertical_unfold(A):
    """
    For a 3D tensor A(a,i,b), we unfold like: A(ia,b)
    """
    S = A.shape
    return A.permute(1, 0, 2).reshape(S[0] * S[1], S[2])


def vertical_refold(Au, i: int):
    """
    Assuming that A was a 3D tensor that was vertically unfolded to Au, this
    returns the original 3D tensor A. Must suply the unfolded tensor Au and
    the length of the central dimension i
    """
    S = Au.shape
    r1 = S[0] // i
    return Au.reshape(i, r1, S[1]).permute(1, 0, 2)


def as_std_core(A):
    """
    Convert a 4D core A(a,i1,i2,b) to 3D like: A(a,i1i2,b)
    """
    S = A.shape
    return A.permute(1, 0, 2).reshape(S[0], S[1] * S[2], S[3])


def mode_1_prod(A, B, adjoint_b=False):
    """
    For a 3D tensor A(r1,i,r2) and B(r3,r1) a matrix, this implements
    the mode 1 product using folds and matrix multiplies only, so that
    we return C(r3,i,r2) = sum_{r1}{A*B}
    """
    S = A.shape
    if adjoint_b:
        return horizontal_refold(B.transpose(-2, -1).conj() @ horizontal_unfold(A), S[1])
    else:
        return horizontal_refold(B @ horizontal_unfold(A), S[1])


def mode_3_prod(A, B, adjoint_b=True):
    """
    For a 3D tensor A(r1,i,r2) and B(r3,r2) a matrix, this implements
    the mode 3 product using folds and matrix multiplies only, so that
    we return C(r1,i,r3) = sum_{r1}{A*B}
    """
    S = A.shape
    if adjoint_b:
        return vertical_refold(vertical_unfold(A) @ B.transpose(-2, -1).conj(), S[1])
    else:
        return vertical_refold(vertical_unfold(A) @ B, S[1])


def mode_2_prod(A, B, adjoint_b=False):
    """
    For a 3D tensor A(r1,i,r2) and B(r3,i) a matrix, this implements
    the mode 2 product so that  we return C(r1,r3,r2) = sum_{i}{A*B}

    adjoint_b = ??? not sure if this is used
    """
    S = A.shape
    uA = A.permute(1, 0, 2).reshape(S[1], S[0] * S[2])
    if adjoint_b:
        r = B.transpose(-2, -1).conj() @ uA
    else:
        r = B @ uA
    return r.reshape(-1, S[0], S[2]).permute(1, 0, 2)


def lq(A):
    qr = torch.linalg.qr(A.transpose(-2, -1))
    return qr[1].transpose(-2, -1), qr[0].transpose(-2, 1)


def insert_into_tuple(y, i, x):
    return (*x[:i], y, *x[i + 1:])


def left_orthogonalize(cores):
    """
    cores is a tuple of tensors
    """
    N = len(cores)
    rprev = torch.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for i in range(N):
        core = cores[i]  # python loop should allow indexing like this
        core_mode_2 = core.shape[1]
        new_core = mode_1_prod(core, rprev)
        if i < N - 1:
            unfolded_core = vertical_unfold(new_core)
            z = torch.linalg.qr(unfolded_core)
            q = z[0]
            rprev = z[1]
            new_core = vertical_refold(q, core_mode_2)
        cores = insert_into_tuple(new_core, i, cores)
    return cores


def right_orthogonalize(cores):
    """
    cores is a tuple of tensors
    """
    N = len(cores)
    lprev = torch.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for i in range(N - 1, -1, -1):
        core = cores[i]
        core_mode_2 = core.shape[1]
        new_core = mode_3_prod(core, lprev)
        if i > 0:
            unfolded_core = horizontal_unfold(new_core)
            z = torch.linalg.qr(unfolded_core.transpose(-2, -1).conj())
            lprev = z[1]
            q = z[0].transpose(-2, -1).conj()
            new_core = horizontal_refold(q, core_mode_2)
        cores = insert_into_tuple(new_core, i, cores)

    return cores


def right_to_left_compression(cores, max_rank: int):
    """
    We assume that cores have already been orthogonalized left to right. This
    is the backwards pass to round down the cores.
    """
    N = len(cores)
    proj_prev = torch.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for i in range(N - 1, -1, -1):
        core = cores[i]
        core_mode_2 = core.shape[1]
        new_core = mode_3_prod(core, proj_prev, adjoint_b=False)
        if i > 0:
            unfolded_core = horizontal_unfold(new_core)
            z = torch.linalg.svd(unfolded_core, full_matrices=False)
            local_rank = min(z.S.shape[0], max_rank)
            proj_prev = z.S[:local_rank][None, :] * z.U[:, :local_rank]
            new_core = horizontal_refold(z.Vh[:local_rank, :], core_mode_2)
        cores = insert_into_tuple(new_core, i, cores)

    return cores


def left_to_right_compression(cores, max_rank: int):
    """
    We assume that cores have already been orthogonalized right to left. This
    is the backwards pass to round down the cores.
    """
    N = len(cores)
    proj_prev = torch.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for i in range(N - 1):
        core = cores[i]
        core_mode_2 = core.shape[1]
        new_core = mode_1_prod(core, proj_prev, adjoint_b=True)
        if i < N - 1:
            unfolded_core = vertical_unfold(new_core)
            z = torch.linalg.svd(unfolded_core, full_matrices=False)
            local_rank = min(z.S.shape[0], max_rank)
            proj_prev = (z.S[:local_rank][:, None] * z.Vh[:local_rank, :]).transpose(-1, -2).conj()
            new_core = vertical_refold(z.U[:, :local_rank], core_mode_2)
        cores = insert_into_tuple(new_core, i, cores)

    return cores


def tt_to_dense(cores):
    N = len(cores)
    if N > 1:
        r = torch.tensordot(cores[-2], cores[-1], [[-1], [0]])
        for k in range(N - 3, -1, -1):
            r = torch.tensordot(cores[k], r, [[-1], [0]])
        r = r.squeeze(0).squeeze(-1)
    else:
        r = cores[-1].squeeze(0)
    return r


def tt_to_dense_einsum(cores):
    """
    an alternative implementation
    """
    N = len(cores)
    if N > 1:
        r = torch.einsum('ijk,kmn->ijmn', cores[-2], cores[-1])
        for k in range(N - 3, -1, -1):
            r = torch.einsum('ijk,k...->ij...', cores[k], r)
        r = r.squeeze(0).squeeze(-1)
    else:
        r = cores[-1].squeeze(0)
    return r


def batch_tt_to_dense(cores):
    N = len(cores)
    if N > 1:
        r = torch.einsum('bijk,bk...->bij...', cores[-2], cores[-1])  # [B,r1,i2,r2] @ [B,r2,i1,1] -> [B,r1,i2,i1,1]
        for k in range(N - 3, -1, -1):
            r = torch.einsum('bijk,bk...->bij...', cores[k], r)  # [B,rm,ik,rk] @ [B,rk,...] -> [B,ik,...]
        r = r.squeeze(1).squeeze(-1)
    else:
        r = cores[-1].squeeze(0)
    return r


def rkr_to_dense(matrices):
    N = len(matrices)
    P = matrices[0].shape[0]
    shapes = [m.shape[-1] for m in matrices]
    if N > 1:
        r = torch.reshape(torch.einsum('ij,im->ijm', matrices[-2], matrices[-1]), [P, -1])
        for k in range(N - 3, -1, -1):
            r = torch.einsum('ij,im->ijm', matrices[k], r).reshape(P, -1)
        r = r.reshape(P, *shapes)
    else:
        r = matrices[-1]
    return r


def cp_to_dense(matrices):
    r = rkr_to_dense(matrices)
    return r.sum(0)


def tt_rkr_inner_dense(cores, matrices):
    B = rkr_to_dense(matrices)
    A = tt_to_dense(cores)
    return torch.tensordot(A, B, [list(range(len(cores))), list(range(1, len(cores) + 1))])


def mode_2_prod_rotated(A, B, adjoint_a=False):
    """
    For a 3D tensor A(r1,i,r2) and B(r3,i) a matrix, this implements
    the mode 2 product so that  we return C(r3,r1,r2) = sum_{i}{A*B}

    Note that we change the order of dimensions in this case (hence
    the name rotated). Now r3 becomes a "batch dim". This is useful
    for subsequent operations
    """
    S = A.shape
    uA = A.permute(1, 0, 2).reshape(S[1], S[0] * S[2])
    if adjoint_a:
        r = B.transpose(-2, -1).conj() @ uA
    else:
        r = B @ uA
    return r.reshape(-1, S[0], S[2])


def tt_rkr_inner(cores, matrices):
    """
    Inner product between a tensor train and a RKR, i.e. a batch of kronecker'd vectors.
    The components of the RKR are matrices ranged like p x i_k, where p is shared over
    all matrices.
    """
    N = len(cores)
    for k in range(N):
        Γ = mode_2_prod_rotated(cores[k], matrices[k])
        cores = insert_into_tuple(Γ, k, cores)
    # now cores will all share a 0th dim of length p.
    if N > 1:
        r = cores[-2] @ cores[-1]
        for k in range(N - 3, -1, -1):
            r = cores[k] @ r
    else:
        r = cores[-1]
    return r.squeeze(-2).squeeze(-1)


def batched_mode_2_product(A, B):
    """
    For a batched 3D tensor A(N,r1,i,r2) and a batched vector (aka matrix) B(N,i),
    this implements the mode 2 product so that we return C(N,r1,r2) = sum_{i}{A*B}
    """
    S = A.shape
    N = S[0]
    r1 = S[1]
    r2 = S[3]
    i = S[2]
    uA = A.permute(0, 2, 1, 3).reshape(N, i, r1 * r2)  # [N,i,r1*r2]
    r = (B[:, None, :] @ uA).squeeze(-2)  # [N,1,i] @ [N,i,r1*r2] -> [N,1,r1*r2] - > [N,r1*r2]
    return r.reshape(N, r1, r2)


def btt_rkr_inner(cores, matrices):
    """
    Innder product between a batched tensor train and a RKR, i.e. a batch of kronecker'd vectors.
    Essentially, this implements a separate tt_kron inner product for each batch dim. Batched
    tensor train cores have shape [B,r1,i,r2] instead of [r1,i,r2]
    """
    N = len(cores)
    for k in range(N):
        Γ = batched_mode_2_product(cores[k], matrices[k])
        cores = insert_into_tuple(Γ, k, cores)
    # now cores will all share a 0th dim of length p.
    if N > 1:
        r = cores[-2] @ cores[-1]
        for k in range(N - 3, -1, -1):
            r = cores[k] @ r
    else:
        r = cores[-1]
    return r.squeeze(-2).squeeze(-1)


def tt_plus_tt(traina, trainb):
    L = len(traina)
    assert len(trainb) == L
    cores_out = []
    for k, (a, b) in enumerate(zip(traina, trainb)):
        r1 = a.shape[0]
        r2 = a.shape[2]
        i = a.shape[1]
        if k == 0:
            cores_out.append(torch.zeros(1, i, 2 * r2))
            cores_out[-1][:, :, :r2] = a
            cores_out[-1][:, :, r2:] = b
        elif k == L - 1:
            cores_out.append(torch.zeros(2 * r1, i, 1))
            cores_out[-1][:r1, :, :] = a
            cores_out[-1][r1:, :, :] = b
        else:
            cores_out.append(torch.zeros(2 * r1, i, 2 * r2))
            cores_out[-1][:r1, :, :r2] = a
            cores_out[-1][r1:, :, r2:] = b
    return cores_out


def batched_tt_plus_tt(traina, trainb):
    L = len(traina)
    assert len(trainb) == L
    cores_out = []
    for k, (a, b) in enumerate(zip(traina, trainb)):
        B = a.shape[0]
        r1 = a.shape[1]
        r2 = a.shape[3]
        i = a.shape[2]
        if k == 0:
            cores_out.append(torch.zeros(B, 1, i, 2 * r2))
            cores_out[-1][:, :, :, :r2] = a
            cores_out[-1][:, :, :, r2:] = b
        elif k == L - 1:
            cores_out.append(torch.zeros(B, 2 * r1, i, 1))
            cores_out[-1][:, :r1, :, :] = a
            cores_out[-1][:, r1:, :, :] = b
        else:
            cores_out.append(torch.zeros(B, 2 * r1, i, 2 * r2))
            cores_out[-1][:, :r1, :, :r2] = a
            cores_out[-1][:, r1:, :, r2:] = b
    return cores_out


def tt_hadamard(traina, trainb):
    L = len(traina)
    assert len(trainb) == L
    cores_out = []
    for a, b in zip(traina, trainb):
        r1 = a.shape[0]
        r2 = a.shape[2]
        i = a.shape[1]
        cores_out.append(torch.einsum('ijk,ljm->jilkm', a, b).reshape(i, r1 * r1, r2 * r2).permute(1, 0, 2))
    return cores_out


def batched_tt_hadamard(traina, trainb):
    L = len(traina)
    assert len(trainb) == L
    cores_out = []
    for a, b in zip(traina, trainb):
        r1 = a.shape[1]
        r2 = a.shape[3]
        i = a.shape[2]
        B = a.shape[0]
        cores_out.append(torch.einsum('bijk,bljm->bjilkm', a, b).reshape(B, i, r1 * r1, r2 * r2).permute(0, 2, 1, 3))
    return cores_out

