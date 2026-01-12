# Paper-to-Code Mapping Guide

This document maps equations and concepts from the DPVO/DPV-SLAM papers directly to code locations.

## DPVO Paper (NeurIPS 2023)

### Equation 1: Patch Representation

**Paper:**
```
P_k = (x, y, 1, d)ᵀ    where x,y,d ∈ ℝ^(1×p²)
```

**Code:** `dpvo/patchgraph.py` (lines ~30-50)
```python
class PatchGraph:
    def __init__(self, ...):
        # Patches stored as [N, M, 3, 3, 3]
        # Last dimension: [x, y, 1, d]
        self.patches_ = torch.zeros(N, M, 3, 3, 3, **kwargs)
```

**Code:** `dpvo/dpvo.py` (line ~427)
```python
# Depth initialization
patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
# patches[:,:,2] is the inverse depth component
```

---

### Equation 2: Patch Reprojection

**Paper:**
```
P'_kj ∼ K·T_j·T_i^(-1)·K^(-1)·P_k
```

**Code:** `dpvo/dpvo.py::reproject()` (lines ~289-313)
```python
def reproject(self, indicies=None):
    """
    Reproject patches from source frame to target frame

    Implements Equation 2 from the DPVO paper:
    P'_kj ∼ K·T_j·T_i^(-1)·K^(-1)·P_k
    """
    (ii, jj, kk) = indicies if indicies is not None else \
                   (self.pg.ii, self.pg.jj, self.pg.kk)
    coords = pops.transform(SE3(self.poses), self.patches,
                           self.intrinsics, ii, jj, kk)
    return coords.permute(0, 1, 4, 2, 3).contiguous()
```

**Implementation:** `dpvo/projective_ops.py::transform()`
```python
def transform(poses, patches, intrinsics, ii, jj, kk):
    # T_j^(-1) · T_i
    rel_pose = poses[jj] * poses[ii].inv()

    # Unproject: K^(-1) · P_k (pixel → 3D)
    X = intrinsics.unproject(patches[kk])

    # Transform: T_j^(-1) · T_i · X
    X = rel_pose * X

    # Project: K · X (3D → pixel)
    coords = intrinsics.project(X)

    return coords
```

---

### Equation 4: Correlation Volume

**Paper:**
```
C(u,v,α,β) = ⟨g(u,v), f(P'(u,v) + Δ_αβ)⟩
```

**Code:** `dpvo/dpvo.py::corr()` (lines ~257-287)
```python
def corr(self, coords, indicies=None):
    """
    Compute correlation volumes for patch matching

    This implements Equation 4 from the DPVO paper:
    C(u,v,α,β) = ⟨g(u,v), f(P'(u,v) + Δ_αβ)⟩
    """
    ii, jj = indicies if indicies is not None else \
             (self.pg.kk, self.pg.jj)
    ii1 = ii % (self.M * self.pmem)
    jj1 = jj % (self.mem)

    # Level 1: 1/4 resolution (radius=3 → 7x7 grid)
    corr1 = altcorr.corr(self.gmap, self.pyramid[0],
                         coords / 1, ii1, jj1, 3)
    # Level 2: 1/16 resolution
    corr2 = altcorr.corr(self.gmap, self.pyramid[1],
                         coords / 4, ii1, jj1, 3)

    return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
```

**CUDA Implementation:** `dpvo/altcorr/correlation.cu`
```cuda
// For each (u,v) in the 3×3 patch
// For each (α,β) in the 7×7 grid
// Compute: dot_product(patch_features[u,v],
//                      frame_features[P'(u,v) + Δ_αβ])

__global__ void corr_kernel(
    const float* __restrict__ fmap,    // frame features
    const float* __restrict__ gmap,    // patch features
    const float* __restrict__ coords,  // reprojections P'
    float* __restrict__ corr,          // output
    int r)                              // radius (3 for 7×7)
{
    // Sample 7×7 grid around coords
    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            // Bilinear interpolation
            float2 pt = coords[...] + make_float2(dx, dy);
            float* f_sampled = sample_bilinear(fmap, pt);

            // Dot product
            corr[...] = dot(gmap[...], f_sampled);
        }
    }
}
```

---

### Equation 6: Bundle Adjustment Objective

**Paper:**
```
argmin  Σ ||Π[T_j^(-1)·T_i·Π^(-1)(P_k)] - I_kj||²_Σ
 T,d   edges

where I_kj = P'_kj + δ_kj  (target)
      Σ_kj = diag(w_kj)     (weight)
```

**Code:** `dpvo/dpvo.py::update()` (lines ~428-506)
```python
def update(self):
    # Step 1: Reproject patches
    coords = self.reproject()

    # Step 2: Compute correlation
    corr = self.corr(coords)

    # Step 3: Network predicts corrections
    self.pg.net, (delta, weight, _) = \
        self.network.update(self.pg.net, ctx, corr, ...)

    # Step 4: Compute targets
    # I_kj = P'_kj + δ_kj
    target = coords[...,self.P//2,self.P//2] + delta.float()

    # Step 5: Bundle Adjustment
    # Minimize: ||Π[...] - target||²_weight
    fastba.BA(self.poses, self.patches, self.intrinsics,
              target, weight, lmbda,
              self.pg.ii, self.pg.jj, self.pg.kk,
              t0, self.n, M=self.M, iterations=2)
```

**CUDA Implementation:** `dpvo/fastba/ba.cu`
```cuda
// Gauss-Newton solver for:
// min Σ ||reproject(T_j, T_i, P_k) - target||²_weight

// For 2 iterations:
for (int iter = 0; iter < 2; iter++) {
    // Linearize residuals
    residual = reproject(poses, patches) - target;
    jacobian = d_reproject / d_poses;

    // Build Hessian: H = JᵀΣJ
    // Build gradient: b = JᵀΣr

    // Solve: H·Δx = -b
    // Use Schur complement to eliminate depth variables

    // Update:
    poses += SE3::exp(Δx_poses);
    depths += Δx_depths;
}
```

---

### Section 3.1: Update Operator

**Paper:** "The update operator is a recurrent module with:
1. Temporal 1D convolutions
2. SoftMax Aggregation
3. Transition Block
4. Factor Head"

**Code:** `dpvo/net.py::Update::forward()` (lines ~125-174)
```python
def forward(self, net, inp, corr, flow, ii, jj, kk):
    # Inject correlation and context
    net = net + inp + self.corr(corr)
    net = self.norm(net)

    # 1D Temporal Convolutions
    ix, jx = fastba.neighbors(kk, jj)  # Find t-1, t+1
    net = net + self.c1(net[:,ix])     # Backward
    net = net + self.c2(net[:,jx])     # Forward

    # SoftMax Aggregation (Message Passing)
    net = net + self.agg_kk(net, kk)           # Same patch
    net = net + self.agg_ij(net, ii*12345+jj)  # Same frame pair

    # Transition Block (GRU)
    net = self.gru(net)

    # Factor Heads
    delta = self.d(net)   # Flow corrections
    weight = self.w(net)  # Confidence weights

    return net, (delta, weight, None)
```

---

### Section 3: Patch Extraction

**Paper:** "We randomly sample keypoints, which works surprisingly well."

**Code:** `dpvo/extractor.py::Patchifier::__patches_from_centroids()`
```python
def __patches_from_centroids(self, centroids, ...):
    # centroids: [M, 2] random 2D locations

    # Extract 3×3 patch features around each centroid
    for u in range(-1, 2):
        for v in range(-1, 2):
            # Bilinear sample at (x+u, y+v)
            patch_features[u,v] = bilinear_sample(
                fmap, centroids + (u, v))

    return patch_features  # [M, 128, 3, 3]
```

**Random Sampling:** `dpvo/extractor.py::random_centroids()`
```python
def random_centroids(H, W, M):
    """Sample M random 2D locations"""
    x = torch.randint(0, W, (M,))
    y = torch.randint(0, H, (M,))
    return torch.stack([x, y], dim=-1)
```

---

## DPV-SLAM Paper (arXiv 2024)

### Section 3.2: Proximity Loop Closure

**Paper:** "Add edges from old patches to recent frames using camera proximity."

**Code:** `dpvo/patchgraph.py::edges_loop()`
```python
def edges_loop(self):
    """Generate loop closure edges based on proximity"""
    ii_loop = []
    jj_loop = []

    # For each recent frame j
    for j in range(n - window, n):
        pose_j = self.poses_[j]

        # Find old frames i within distance threshold
        for i in range(0, n - max_age):
            pose_i = self.poses_[i]

            # Check proximity
            dist = (pose_i.translation() -
                   pose_j.translation()).norm()

            if dist < proximity_thresh:
                # Add edges: patches from i → frame j
                patches_i = range(i*M, (i+1)*M)
                ii_loop.extend(patches_i)
                jj_loop.extend([j] * M)

    return torch.tensor(ii_loop), torch.tensor(jj_loop)
```

---

### Figure 2: Uni-directional Edges

**Paper:** "Unlike DROID which uses bi-directional edges, we only create edges from old patches to new frames. This saves 15x memory!"

**Code:** `dpvo/dpvo.py::__edges_forw()` and `__edges_back()`
```python
def __edges_forw(self):
    """Forward edges: recent patches → current frame"""
    r = self.cfg.PATCH_LIFETIME  # 13 frames

    # Patches from last r frames
    t0 = self.M * max((self.n - r), 0)
    t1 = self.M * max((self.n - 1), 0)
    patches = torch.arange(t0, t1)

    # Current frame
    frames = torch.arange(self.n-1, self.n)

    # Create edges: patches → frames
    return flatmeshgrid(patches, frames)

def __edges_back(self):
    """Backward edges: current patches → recent frames"""
    r = self.cfg.PATCH_LIFETIME

    # Current patches
    t0 = self.M * (self.n - 1)
    t1 = self.M * self.n
    patches = torch.arange(t0, t1)

    # Recent frames
    frames = torch.arange(max(self.n-r, 0), self.n)

    return flatmeshgrid(patches, frames)
```

**Memory Comparison:**
```
Bi-directional (DROID):
  Store features for ALL frames
  Memory: 9.1 GB / 1K frames

Uni-directional (DPVO):
  Store features only for destination frames
  Memory: 0.6 GB / 1K frames

Savings: 15x!
```

---

### Section 3.3: Classical Loop Closure

**Paper:** Uses DBoW2 + DISK + LightGlue + Sim(3) alignment

**Code:** `dpvo/loop_closure/long_term.py::attempt_loop_closure()`
```python
def attempt_loop_closure(self, n):
    # Step 1: Image retrieval (DBoW2)
    candidates = self.retrieval.query(self.features[n])

    # Step 2: Match keypoints (DISK + LightGlue)
    for cand_idx in candidates:
        matches = self.matcher.match(
            self.keypoints[n],
            self.keypoints[cand_idx])

        if len(matches) < min_matches:
            continue

        # Step 3: Structure-only BA
        # Triangulate 3D points from matches
        pts_3d_n = triangulate(matches, n, n-1, n-2)
        pts_3d_c = triangulate(matches, cand_idx, ...)

        # Step 4: Sim(3) alignment (RANSAC + Umeyama)
        sim3 = align_point_clouds(pts_3d_n, pts_3d_c)

        if sim3.inliers < threshold:
            continue

        # Step 5: Add to pose graph
        self.loop_constraints.append((n, cand_idx, sim3))
```

**Pose Graph Optimization:** `dpvo/loop_closure/optim_utils.py`
```python
def pose_graph_optimization(poses, loop_constraints):
    """
    Minimize:
      Σ ||log(ΔS_{i,i+1}^(-1)·S_i^(-1)·S_{i+1})||² +
      Σ ||log(ΔS_jk^loop·S_j^(-1)·S_k)||²
    """
    for iter in range(10):
        # Smoothness residuals
        for i in range(N-1):
            r_i = log_sim3(
                S[i+1] * S[i].inv() * delta_S[i,i+1].inv())

        # Loop residuals
        for (j, k, delta_S_jk) in loop_constraints:
            r_jk = log_sim3(
                S[k] * S[j].inv() * delta_S_jk)

        # Levenberg-Marquardt update
        H = build_hessian(r_i, r_jk)
        b = build_gradient(r_i, r_jk)

        delta_S = solve(H, b)
        S += delta_S
```

---

## Training (DPVO Paper Section 4)

### Loss Function

**Paper:**
```
L = 10·L_pose + 0.1·L_flow

L_pose = Σ ||log((G_i^(-1)·G_j)^(-1)·(T_i^(-1)·T_j))||
L_flow = min over p×p ||flow_pred - flow_gt||
```

**Code:** `train.py` (approximate location)
```python
def compute_loss(poses_pred, poses_gt, flow_pred, flow_gt):
    # Pose loss
    loss_pose = 0
    for i, j in frame_pairs:
        # Ground truth relative pose
        G_rel = poses_gt[j] * poses_gt[i].inv()

        # Predicted relative pose
        T_rel = poses_pred[j] * poses_pred[i].inv()

        # SE(3) error
        error = (G_rel.inv() * T_rel).log()
        loss_pose += error.norm()

    # Flow loss (minimum over patch)
    loss_flow = 0
    for edge in edges:
        # flow_pred: [3, 3, 2]
        # flow_gt: [3, 3, 2]
        errors = (flow_pred - flow_gt).norm(dim=-1)
        loss_flow += errors.min()  # Best match in 3×3

    return 10 * loss_pose + 0.1 * loss_flow
```

---

## Key Data Structures

### Patch Graph Structure

**Code:** `dpvo/patchgraph.py`
```python
class PatchGraph:
    def __init__(self):
        # Camera poses (SE3)
        self.poses_ = torch.zeros(N, 7)  # [tx,ty,tz, qx,qy,qz,qw]

        # Patches (fronto-parallel planes)
        self.patches_ = torch.zeros(N, M, 3, 3, 3)
        # Last dim: [x, y, 1, d] for each pixel in 3×3

        # Edges (ii → jj via patch kk)
        self.ii = torch.tensor([])  # Source frame
        self.jj = torch.tensor([])  # Dest frame
        self.kk = torch.tensor([])  # Patch index

        # Factors (predictions from network)
        self.target = torch.zeros(1, E, 2)  # Where patches should be
        self.weight = torch.zeros(1, E, 2)  # Confidence weights

        # Hidden states (for recurrent network)
        self.net = torch.zeros(1, E, 384)
```

---

## CUDA Kernels

### Correlation (`dpvo/altcorr/`)

**Purpose:** Compute dot products between patch features and frame features

**Input:**
- `gmap`: Patch features [P, 128, 3, 3]
- `fmap`: Frame features [N, 128, H, W]
- `coords`: Reprojected locations [E, 3, 3, 2]
- `r`: Radius (3 for 7×7 grid)

**Output:**
- `corr`: Correlation volume [E, 3, 3, 7, 7]

**Algorithm:**
```
for each edge e:
  for each pixel (u,v) in 3×3 patch:
    for each offset (dx,dy) in 7×7 grid:
      pt = coords[e,u,v] + (dx,dy)
      f = bilinear_sample(fmap[e.j], pt)
      corr[e,u,v,dx,dy] = dot(gmap[e.k,u,v], f)
```

### Bundle Adjustment (`dpvo/fastba/`)

**Purpose:** Optimize poses and depths to minimize reprojection error

**Input:**
- `poses`: [N, 7] SE3 poses
- `patches`: [N*M, 3, 3, 3] patch locations
- `intrinsics`: [N, 4] camera parameters
- `target`: [E, 2] where patches should reproject to
- `weight`: [E, 2] confidence weights

**Output:**
- Updated `poses` and `patches`

**Algorithm:**
```
for iteration in range(2):
  // Compute residuals
  for each edge (i,j,k):
    r = reproject(poses[j], poses[i], patches[k]) - target

  // Build Hessian (block structure)
  H_TT = JT_pose^T @ W @ JT_pose
  H_Td = JT_pose^T @ W @ JT_depth
  H_dd = JT_depth^T @ W @ JT_depth

  // Schur complement (eliminate depths)
  H_schur = H_TT - H_Td @ inv(H_dd) @ H_Td^T

  // Solve for pose updates
  delta_T = solve(H_schur, b_T)

  // Back-substitute for depths
  delta_d = inv(H_dd) @ (b_d - H_Td^T @ delta_T)

  // Update (on SE3 manifold)
  poses = SE3::exp(delta_T) * poses
  depths = depths + delta_d
```

---

This mapping should help you understand exactly where each paper concept is implemented in the code!
