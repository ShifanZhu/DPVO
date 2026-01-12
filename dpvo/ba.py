import torch
from torch_scatter import scatter_sum

from . import fastba
from . import lietorch
from .lietorch import SE3

from .utils import Timer

from . import projective_ops as pops


class CholeskySolver(torch.autograd.Function):
    """
    Differentiable Cholesky-based linear system solver.

    Solves Hx = b using Cholesky decomposition with custom backward pass.
    Gracefully handles decomposition failures during training by returning zeros.
    """
    @staticmethod
    def forward(ctx, H, b):
        """
        Solve Hx = b using Cholesky decomposition.

        Args:
            H: Symmetric positive definite matrix
            b: Right-hand side vector

        Returns:
            Solution x, or zeros if decomposition fails
        """
        # Use cholesky_ex which doesn't crash on failure
        U, info = torch.linalg.cholesky_ex(H)

        # If decomposition failed (info != 0), return zeros to continue training
        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        # Solve using Cholesky factor U
        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        """
        Custom backward pass for Cholesky solver.

        Implements implicit differentiation: if Hx = b, then dL/dH and dL/db
        can be computed from dL/dx without re-solving the system.
        """
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        # Solve for gradient w.r.t. b
        dz = torch.cholesky_solve(grad_x, U)
        # Compute gradient w.r.t. H using implicit differentiation
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz


# Utility functions for scatter operations

def safe_scatter_add_mat(A, ii, jj, n, m):
    """
    Safely scatter-add matrix elements with bounds checking.

    Args:
        A: Values to scatter [batch, num_elements, ...]
        ii, jj: Row and column indices
        n, m: Matrix dimensions

    Returns:
        Scattered matrix of shape [batch, n*m, ...]
    """
    # Only include valid indices within bounds
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    """
    Safely scatter-add vector elements with bounds checking.

    Args:
        b: Values to scatter [batch, num_elements, ...]
        ii: Indices
        n: Vector length

    Returns:
        Scattered vector of shape [batch, n, ...]
    """
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)


# Retraction operators for manifold optimization

def disp_retr(disps, dz, ii):
    """
    Apply retraction (update) to inverse depth values.

    Args:
        disps: Current inverse depth values
        dz: Depth updates
        ii: Indices mapping updates to depth values

    Returns:
        Updated inverse depths
    """
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

def pose_retr(poses, dx, ii):
    """
    Apply retraction (update) to SE3 poses on the manifold.

    Args:
        poses: Current SE3 poses
        dx: Tangent space updates (se3 algebra)
        ii: Indices mapping updates to poses

    Returns:
        Updated SE3 poses
    """
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(A, B):
    """
    Multiply two block-structured matrices.

    Efficiently multiplies matrices with block structure by reshaping,
    performing standard matrix multiply, and reshaping back.

    Args:
        A: Block matrix of shape [batch, n1, m1, p1, q1]
        B: Block matrix of shape [batch, n2, m2, p2, q2]
           where m1*q1 must equal n2*p2 for matrix multiplication

    Returns:
        Block matrix product of shape [batch, n1, m2, p1, q2]
    """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    # Reshape blocks into standard matrix format
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    # Standard matrix multiply, then reshape back to block format
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

def block_solve(A, B, ep=1.0, lm=1e-4):
    """
    Solve block-structured linear system AX = B.

    Args:
        A: Block matrix [batch, n1, m1, p1, q1] (must be square: n1*p1 = m1*q1)
        B: Block matrix [batch, n2, m2, p2, q2]
        ep: Epsilon for numerical stability (added to diagonal)
        lm: Levenberg-Marquardt damping factor

    Returns:
        Solution X as block matrix [batch, n1, m2, p1, q2]
    """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    # Reshape to standard matrix format
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    # Add damping for numerical stability (Levenberg-Marquardt style)
    A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    # Solve using custom Cholesky solver with backprop support
    X = CholeskySolver.apply(A, B)
    # Reshape back to block format
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    """
    Debug utility to visualize block matrix structure.

    Args:
        A: Block matrix [batch, n1, m1, p1, q1]
    """
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()

def BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=100.0, PRINT=False, fixedp=1, structure_only=False):
    """
    Perform bundle adjustment to jointly optimize camera poses and 3D structure.

    Implements sparse bundle adjustment using Schur complement trick to efficiently
    solve the normal equations. Optimizes SE3 camera poses and inverse depths.

    Args:
        poses: Camera poses as SE3 [batch, num_frames, 7]
        patches: Image patches with depth [batch, num_patches, 3, patch_size, patch_size]
        intrinsics: Camera intrinsics [batch, num_frames, 4] (fx, fy, cx, cy)
        targets: Target 2D coordinates for reprojection [batch, num_edges, 2]
        weights: Edge weights for robust optimization [batch, num_edges, 2]
        lmbda: Damping parameter for inverse depth (scalar or per-point)
        ii, jj, kk: Edge indices (source frame, target frame, patch)
        bounds: Image bounds [x_min, y_min, x_max, y_max] for validity checking
        ep: Epsilon for pose damping
        PRINT: Whether to print residual statistics
        fixedp: Number of fixed poses (first N poses won't be optimized)
        structure_only: If True, only optimize depth (freeze poses)

    Returns:
        Tuple of (optimized_poses, optimized_patches)
    """
    b = 1
    # Number of frames to optimize
    n = max(ii.max().item(), jj.max().item()) + 1

    # Project patches to target frames and compute Jacobians
    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    # Compute reprojection residuals (target - prediction)
    p = coords.shape[3]
    r = targets - coords[...,p//2,p//2,:]

    # Filter out edges with very large residuals (outliers)
    v *= (r.norm(dim=-1) < 250).float()

    # Check if projections are within image bounds
    in_bounds = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    # Mask invalid edges (out of bounds or large residuals)
    v *= in_bounds.float()

    if PRINT:
        # Print mean reprojection error for debugging
        print((r * v[...,None]).norm(dim=-1).mean().item())

    # Apply validity mask to residuals and weights
    r = (v[...,None] * r).unsqueeze(dim=-1)
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    # Compute weighted Jacobian transposes for normal equations
    wJiT = (weights * Ji).transpose(2,3)  # w.r.t. source pose
    wJjT = (weights * Jj).transpose(2,3)  # w.r.t. target pose
    wJzT = (weights * Jz).transpose(2,3)  # w.r.t. depth

    # Build blocks of Hessian approximation (J^T W J)
    # B blocks: pose-pose interactions
    Bii = torch.matmul(wJiT, Ji)  # Source pose self-interaction
    Bij = torch.matmul(wJiT, Jj)  # Source-target pose interaction
    Bji = torch.matmul(wJjT, Ji)  # Target-source pose interaction
    Bjj = torch.matmul(wJjT, Jj)  # Target pose self-interaction

    # E blocks: pose-depth interactions
    Eik = torch.matmul(wJiT, Jz)  # Source pose - depth
    Ejk = torch.matmul(wJjT, Jz)  # Target pose - depth

    # Right-hand side vectors (J^T W r)
    vi = torch.matmul(wJiT, r)  # For source poses
    vj = torch.matmul(wJjT, r)  # For target poses

    # Exclude fixed poses from optimization
    ii = ii.clone()
    jj = jj.clone()

    n = n - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    # Get unique depth variables and remap indices
    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    # Assemble pose-pose block of Hessian by scattering edge contributions
    B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    # Assemble pose-depth block of Hessian
    E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1)

    # Assemble depth-depth diagonal block of Hessian
    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    # Assemble pose right-hand side
    v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    # Assemble depth right-hand side
    w = safe_scatter_add_vec(torch.matmul(wJzT, r), kk, m)

    # Apply damping to depth diagonal
    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)

    Q = 1.0 / (C + lmbda)  # Inverse of damped depth diagonal

    ### Solve using Schur complement trick ###
    # Schur complement eliminates depth variables, solving smaller system for poses
    EQ = E * Q[:,None]

    if structure_only or n == 0:
        # Only optimize depth (poses fixed or no poses to optimize)
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        # Solve for both poses and depth using Schur complement
        # Reduced system: (B - E Q E^T) dX = v - E Q w
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX = block_solve(S, y, ep=ep, lm=1e-4)

        # Back-substitute to get depth update: dZ = Q (w - E^T dX)
        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    # Apply updates to patches (depth component)
    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    # Apply updates to poses (if optimizing poses)
    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches
