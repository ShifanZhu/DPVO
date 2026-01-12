# DPVO Architecture Overview

## System-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DPVO System Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

Input: RGB Image [H×W×3] + Intrinsics [fx, fy, cx, cy]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. PATCHIFIER (Feature Extraction)                                  │
│                                                                      │
│    ┌──────────────┐     ┌──────────────┐                           │
│    │ Matching Net │     │ Context Net  │                           │
│    │  (ResNet)    │     │  (ResNet)    │                           │
│    └──────┬───────┘     └──────┬───────┘                           │
│           │                    │                                    │
│           ├─── [128, H/4, W/4] (2-level pyramid)                   │
│           │                    │                                    │
│           │                    └─── [384, H/4, W/4]                │
│           │                                                         │
│    ┌──────▼─────────────────────────────────────┐                  │
│    │  Random Patch Sampling (96 patches)       │                  │
│    └──────┬─────────────────────────────────────┘                  │
│           │                                                         │
│    Outputs:                                                         │
│    • gmap: [96, 128, 3, 3] - matching features                     │
│    • imap: [96, 384] - context features                            │
│    • patches: [96, 3, 3, 3] - (x, y, 1, d)                        │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. PATCH GRAPH (State Management)                                   │
│                                                                      │
│    Stores:                                                          │
│    • Poses: [N, 7] camera poses (x,y,z, qx,qy,qz,qw)              │
│    • Patches: [N×M, 3, 3, 3] patch locations + depths              │
│    • Edges: (ii, jj, kk) connections                               │
│                                                                      │
│    Patch Graph Structure:                                           │
│                                                                      │
│    Frame 0   Frame 1   Frame 2   Frame 3                           │
│      │         │         │         │                                │
│      P₀        P₁        P₂        P₃                              │
│      P₁        P₂        P₃        P₄                              │
│      ...       ...       ...       ...                             │
│      P₉₅       P₉₅       P₉₅       P₉₅                            │
│       \       / \       / \       /                                │
│        \     /   \     /   \     /                                 │
│         Edges (forward + backward + loop)                          │
│                                                                      │
│    Example edges for patch P₁ from Frame 1:                        │
│    • Forward: P₁ → Frame 2, Frame 3 (track into future)           │
│    • Backward: P₁ → Frame 0 (refine past)                         │
│    • Loop: P₁ → Frame 100 (if proximity detected)                 │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. UPDATE OPERATOR (Iterative Refinement)                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ a) Reproject patches using current poses & depths       │       │
│  │    P'_kj = K·T_j·T_i^(-1)·K^(-1)·P_k                   │       │
│  └───────────────────────────┬─────────────────────────────┘       │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ b) Compute Correlation (visual similarity)              │       │
│  │    For each edge (k,j):                                 │       │
│  │    • Sample 7×7 grid around reprojected center          │       │
│  │    • Dot product with patch features                    │       │
│  │    • Two pyramid levels → [E, 2×49×9]                  │       │
│  └───────────────────────────┬─────────────────────────────┘       │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ c) Neural Network (GRU-based)                           │       │
│  │                                                          │       │
│  │    Hidden State [E, 384]                                │       │
│  │         +                                                │       │
│  │    Context Features [E, 384]                            │       │
│  │         +                                                │       │
│  │    Correlation Features [E, 882]                        │       │
│  │         ↓                                                │       │
│  │    1D Conv (temporal)                                   │       │
│  │         ↓                                                │       │
│  │    SoftAgg (message passing)                            │       │
│  │         ↓                                                │       │
│  │    GRU (2× gated residual)                             │       │
│  │         ↓                                                │       │
│  │    Factor Heads:                                        │       │
│  │    • delta: [E, 2] flow corrections                     │       │
│  │    • weight: [E, 2] confidence (0,1)                   │       │
│  └───────────────────────────┬─────────────────────────────┘       │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ d) Bundle Adjustment (optimize T and d)                 │       │
│  │                                                          │       │
│  │    Minimize:                                            │       │
│  │    Σ ||ω_ij(T,P_k) - [P'_kj + δ_kj]||²_Σ              │       │
│  │                                                          │       │
│  │    Using:                                               │       │
│  │    • 2 Gauss-Newton iterations                         │       │
│  │    • Block-sparse Hessian (CUDA)                       │       │
│  │    • Schur complement trick                            │       │
│  │                                                          │       │
│  │    Variables:                                           │       │
│  │    • T: poses [N, 7]                                   │       │
│  │    • d: inverse depths [N×M, 1]                        │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. KEYFRAMING (Remove redundant frames)                            │
│                                                                      │
│    If motion(frame_{t-3}, frame_{t-1}) < 15 pixels:                │
│    • Remove frame_{t-2}                                            │
│    • Store relative pose for interpolation                         │
│    • Shift all arrays                                              │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. LOOP CLOSURE (DPV-SLAM only)                                    │
│                                                                      │
│  ┌────────────────────┐      ┌───────────────────────┐            │
│  │ Proximity LC       │      │ Classical LC (DBoW2)  │            │
│  │ (GPU, main thread) │      │ (CPU, parallel)       │            │
│  └────────────────────┘      └───────────────────────┘            │
│           │                            │                            │
│           ▼                            ▼                            │
│    Detect loops via         Image retrieval +                      │
│    camera proximity         DISK+LightGlue                         │
│           │                            │                            │
│           ▼                            ▼                            │
│    Add uni-directional      Sim(3) alignment                       │
│    edges (old→new)          + RANSAC                               │
│           │                            │                            │
│           ▼                            ▼                            │
│    Global BA                Pose graph                             │
│    (efficient block-        optimization                           │
│     sparse solver)                                                 │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
Output: Poses [N, 7] + Point Cloud [M×N, 3] + Colors
```

## Update Operator Detail

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Update Operator Architecture                       │
│                    (Recurrent Neural Network)                         │
└──────────────────────────────────────────────────────────────────────┘

Input per edge (k,j):
  • Hidden state h_{k,j} [384]
  • Context features ctx_k [384]
  • Correlation features corr_{k,j} [882]

┌────────────────────────────────────────────────────────────────┐
│ Step 1: Inject New Information                                 │
│                                                                 │
│    h ← h + ctx + MLP(corr)                                     │
│    h ← LayerNorm(h)                                            │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 2: 1D Temporal Convolution                                │
│                                                                 │
│    For each edge (k,j), find temporal neighbors:               │
│    • ix = edge with patch k in frame j-1                       │
│    • jx = edge with patch k in frame j+1                       │
│                                                                 │
│    h ← h + MLP1(h[ix])  (backward)                            │
│    h ← h + MLP2(h[jx])  (forward)                             │
│                                                                 │
│    Effect: Propagates info along patch trajectory              │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 3: SoftMax Aggregation (Message Passing)                  │
│                                                                 │
│    a) Patch aggregation (same source patch):                   │
│       h ← h + SoftAgg(h, group_by=kk)                          │
│                                                                 │
│    b) Frame-pair aggregation (same src & dst frame):           │
│       h ← h + SoftAgg(h, group_by=ii*12345+jj)                │
│                                                                 │
│    SoftAgg(x, groups):                                         │
│      w = sigmoid(Linear(x))                                    │
│      out = Σ_{same_group} w·x / Σ_{same_group} w             │
│                                                                 │
│    Effect: Shares info between related edges                   │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 4: Transition Block (GRU)                                 │
│                                                                 │
│    h ← GatedResidual(LayerNorm(h))                            │
│    h ← GatedResidual(LayerNorm(h))                            │
│                                                                 │
│    GatedResidual(x):                                           │
│      z = sigmoid(Linear(x))                                    │
│      r = ReLU(Linear(x))                                       │
│      return z·x + (1-z)·r                                     │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 5: Factor Heads                                           │
│                                                                 │
│    delta = MLP_delta(h)  → [E, 2]                             │
│    weight = sigmoid(MLP_weight(h))  → [E, 2] ∈ (0,1)         │
│                                                                 │
│    Output: (h_new, delta, weight)                              │
└────────────────────────────────────────────────────────────────┘
```

## Bundle Adjustment Detail

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Bundle Adjustment (CUDA)                           │
│                 Minimize Reprojection Error                           │
└──────────────────────────────────────────────────────────────────────┘

Given:
  • Poses T [N, 7]
  • Patches P [N×M, 3, 3, 3]  (x, y, 1, d)
  • Targets I [E, 2]  (where patches should be)
  • Weights Σ [E, 2]  (confidence)
  • Edges (ii, jj, kk)

Objective:
  min Σ_{(i,j,k)} ||ω_ij(T, P_k) - I_{k,j}||²_Σ
  T,d

Where ω_ij projects patch k from frame i to frame j.

┌────────────────────────────────────────────────────────────────┐
│ Algorithm: Gauss-Newton (2 iterations)                          │
│                                                                 │
│ For iter = 1, 2:                                               │
│   1. Linearize residuals around current T, d                   │
│   2. Build Hessian H and gradient b                            │
│   3. Solve H·Δx = -b using block-sparse structure             │
│   4. Update: T ← T ⊕ Δx_T, d ← d + Δx_d                      │
└────────────────────────────────────────────────────────────────┘

Block Structure:

  Variables: [T₀, T₁, ..., T_N, d₀, d₁, ..., d_{N×M}]
                ↑              ↑
            Poses (7-DOF)   Depths (1-DOF)

  Hessian (block-sparse):

         T₀  T₁  ...  T_N │ d₀  d₁  ...
    T₀  [█████████       │           ]
    T₁  [█████████       │           ]
    ... [      █████████ │           ]
    T_N [      █████████ │           ]
    ────┼─────────────────┼─────────────
    d₀  [                │ █         ]
    d₁  [                │   █       ]
    ... [                │     █     ]

  Schur complement trick:
    1. Eliminate depth variables first (cheap)
    2. Solve pose-only system (expensive but smaller)
    3. Back-substitute for depths

┌────────────────────────────────────────────────────────────────┐
│ Two Implementations:                                            │
│                                                                 │
│ • Local BA (eff_impl=False):                                   │
│   - Dense Hessian                                              │
│   - Fast for small problems (<100 poses)                       │
│   - Used for sliding window optimization                       │
│                                                                 │
│ • Global BA (eff_impl=True):                                   │
│   - Block-sparse Hessian                                       │
│   - Efficient for large problems (>1000 poses)                 │
│   - Used for loop closure                                      │
│   - Custom CUDA kernel with sparse indexing                    │
└────────────────────────────────────────────────────────────────┘
```

## Memory Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Memory Organization                              │
└──────────────────────────────────────────────────────────────────────┘

Circular Buffers (wrap around using % operator):

┌─────────────────────────────────────────┐
│ Frame Features (mem = 36 frames)        │
├─────────────────────────────────────────┤
│ fmap1[0:36, 128, H/4, W/4]  → 1/4 res  │
│ fmap2[0:36, 128, H/16, W/16] → 1/16 res │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Patch Features (pmem = 36 frames)       │
├─────────────────────────────────────────┤
│ imap[0:36, 96, 384]    → context        │
│ gmap[0:36, 96, 128, 3, 3] → matching    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Patch Graph (N = 2048 frames max)       │
├─────────────────────────────────────────┤
│ poses[0:N, 7]                           │
│ patches[0:N, 96, 3, 3, 3]               │
│ intrinsics[0:N, 4]                      │
│ colors[0:N, 96, 3]                      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Edges (Dynamic, ~1200 per frame)        │
├─────────────────────────────────────────┤
│ ii: source frame indices [E]            │
│ jj: destination frame indices [E]       │
│ kk: patch indices [E]                   │
│ target: predictions [1, E, 2]           │
│ weight: confidence [1, E, 2]            │
│ net: hidden states [1, E, 384]          │
└─────────────────────────────────────────┘

Total GPU Memory (Default Config):
  • Feature maps: ~800 MB
  • Patch features: ~600 MB
  • Patch graph: ~2 GB
  • Edges & factors: ~1.5 GB
  • Total: ~4.9 GB
```

## Comparison: DPVO vs DROID-SLAM

```
┌────────────────────────────────────────────────────────────────┐
│                     DPVO vs DROID-SLAM                          │
└────────────────────────────────────────────────────────────────┘

Feature          │ DPVO              │ DROID-SLAM
─────────────────┼───────────────────┼──────────────────
Representation   │ Sparse patches    │ Dense flow
                 │ (96 per frame)    │ (all pixels)
─────────────────┼───────────────────┼──────────────────
Feature res.     │ 1/4 (H/4 × W/4)   │ 1/8 (H/8 × W/8)
─────────────────┼───────────────────┼──────────────────
Correlation      │ On-the-fly        │ Pre-computed
                 │ (7×7 grid)        │ (full volumes)
─────────────────┼───────────────────┼──────────────────
Update operator  │ GRU + 1D-conv     │ GRU only
                 │ + Message passing │
─────────────────┼───────────────────┼──────────────────
FPS (average)    │ 60 FPS            │ 40 FPS
─────────────────┼───────────────────┼──────────────────
FPS (worst)      │ 48 FPS            │ 11 FPS
─────────────────┼───────────────────┼──────────────────
Memory           │ 4.9 GB            │ 8.7 GB
─────────────────┼───────────────────┼──────────────────
Accuracy (ATE)   │ Similar or better │ Very good
─────────────────┼───────────────────┼──────────────────
Loop closure     │ Uni-directional   │ Bi-directional
                 │ (memory efficient)│ (memory heavy)
─────────────────┼───────────────────┼──────────────────
Runtime          │ Constant          │ Variable
                 │ (~17ms/frame)     │ (depends on motion)
```

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                             │
└──────────────────────────────────────────────────────────────────────┘

Dataset: TartanAir (synthetic)
  • 50+ environments
  • Indoor + outdoor
  • Varied lighting/weather

┌────────────────────────────────────────────────────────────────┐
│ 1. Sample 15-frame sequence                                     │
│    • First 8 frames: initialization                             │
│    • Next 7 frames: added one-by-one                            │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. Augmentation                                                 │
│    • Random crop & resize                                       │
│    • Color jitter                                               │
│    • Random grayscale                                           │
│    • Random color inversion                                     │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. Forward pass (18 update iterations)                          │
│    • Extract features                                           │
│    • Sample patches randomly                                    │
│    • Run update operator 18 times                               │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. Compute loss                                                 │
│                                                                 │
│    L = 10·L_pose + 0.1·L_flow                                  │
│                                                                 │
│    L_pose = Σ ||log((G_i⁻¹·G_j)⁻¹·(T_i⁻¹·T_j))||             │
│    L_flow = min over 3×3 ||flow_pred - flow_gt||              │
│                                                                 │
│    Detach T, P before each update iteration                     │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. Backpropagation through:                                     │
│    • Update operator (GRU)                                      │
│    • Bundle adjustment (differentiable)                         │
│    • Feature extractors                                         │
└────────────────────────────────────────────────────────────────┘

Training details:
  • 240k iterations (~3.5 days on RTX-3090)
  • Batch size: 1
  • Optimizer: AdamW, LR=8e-5 (linear decay)
  • First 1000 steps: fix poses (structure-only)
```

This architecture achieves state-of-the-art accuracy while running 3x faster than previous methods!
