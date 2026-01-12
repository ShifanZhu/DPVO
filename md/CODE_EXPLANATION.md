# DPVO Code Explanation

This document provides a comprehensive guide to understanding the DPVO codebase and how it relates to the two papers.

## Papers

1. **Deep Patch Visual Odometry (DPVO)** - NeurIPS 2023
   - Introduces sparse patch-based visual odometry
   - Runs 3x faster than DROID-SLAM using 1/3 memory
   - Achieves state-of-the-art accuracy

2. **Deep Patch Visual SLAM (DPV-SLAM)** - arXiv 2024
   - Extends DPVO with loop closure capabilities
   - Two backends: proximity-based and classical (DBoW2)
   - Maintains efficiency while reducing drift

## Core Architecture

### 1. Main System (`dpvo/dpvo.py`)

The `DPVO` class orchestrates the entire pipeline:

```
Input Frame → Patchify → Motion Model → Add Edges → Update+BA → Keyframe → Output
```

#### Key Methods:

**`__init__()`**
- Initializes buffers for patches and features
- Sets up patch graph data structure
- Configures mixed precision (FP16 for speed)
- Memory allocation:
  - `imap_`: Context features [36, 96, 384]
  - `gmap_`: Matching features [36, 96, 128, 3, 3]
  - `fmap1_`, `fmap2_`: Feature pyramid for correlation

**`__call__(tstamp, image, intrinsics)`** - Main entry point
```python
# Step 1: Feature Extraction
fmap, gmap, imap, patches = network.patchify(image)

# Step 2: Pose Initialization (damped linear motion model)
# Predicts new pose from previous motion with damping

# Step 3: Depth Initialization
# Uses median depth from recent patches

# Step 4: Add Edges
# Forward: recent patches → current frame
# Backward: current patches → recent frames
# Loop closure: old patches → current frame (optional)

# Step 5: Optimization
update()  # Network + Bundle Adjustment

# Step 6: Keyframing
keyframe()  # Remove redundant frames
```

**`update()`** - Core optimization loop
```
┌─────────────┐
│  Reproject  │  Transform patches using current poses
└──────┬──────┘
       │
┌──────▼──────┐
│ Correlation │  Compute visual similarity (Eq. 4)
└──────┬──────┘
       │
┌──────▼──────┐
│   Update    │  Neural network predicts:
│  Operator   │  - delta: flow corrections
└──────┬──────┘  - weight: confidence
       │
┌──────▼──────┐
│   Bundle    │  Minimize reprojection error (Eq. 6)
│ Adjustment  │  Optimize T (poses) and d (depths)
└─────────────┘
```

**`corr(coords, indicies)`** - Correlation computation
- Implements Equation 4: `C(u,v,α,β) = ⟨g(u,v), f(P'(u,v) + Δ_αβ)⟩`
- Samples 7×7 grid around reprojected patch
- Uses 2-level pyramid (1/4 and 1/16 resolution)
- Custom CUDA kernel for efficiency

**`reproject(indicies)`** - Geometric reprojection
- Implements Equation 2: `P'_kj = K·T_j·T_i^(-1)·K^(-1)·P_k`
- Projects 3x3 patches from frame i to frame j
- Uses current pose and depth estimates

### 2. Network Architecture (`dpvo/net.py`)

#### VONet Components:

**Update Operator** - Recurrent refinement module
```
Input: Hidden state + Context + Correlation
       ↓
1. Inject correlation features (visual similarity)
       ↓
2. 1D Temporal Convolutions (along patch trajectories)
       ↓
3. SoftMax Aggregation (message passing)
   - agg_kk: patches with same source
   - agg_ij: patches with same frame pair
       ↓
4. Transition Block (2x Gated Residual Units)
       ↓
5. Factor Heads:
   - delta: 2D flow corrections [E, 2]
   - weight: confidence [E, 2] in (0, 1)
```

**Patchifier** - Feature extraction + patch sampling
- ResNet encoder for features
- Random patch sampling (surprisingly effective!)
- Outputs:
  - fmap: Frame features for correlation
  - gmap: Patch matching features (3x3)
  - imap: Context features (384-dim)
  - patches: Initial locations [M, 3, 3, 3]

### 3. Patch Graph (`dpvo/patchgraph.py`)

Bipartite graph connecting patches to frames:

```
Patches (source)  →  Edges  →  Frames (destination)
    P₀, P₁, ...           ↓         F₀, F₁, ...
                    (ii, jj, kk)
```

**Data Structure:**
- `poses_`: Camera poses [N, 7] (x, y, z, qx, qy, qz, qw)
- `patches_`: Patches [N*M, 3, 3, 3] (x, y, depth)
- `ii, jj, kk`: Edge indices (source frame, dest frame, patch)
- `target`: Predicted reprojections [1, E, 2]
- `weight`: Confidence weights [1, E, 2]

### 4. Bundle Adjustment (`dpvo/fastba/`)

Custom CUDA implementation for efficiency:

**Objective (Equation 6):**
```
min  Σ ||ω_ij(T,P_k) - [P'_kj + δ_kj]||²_Σ
T,d

Where:
- T: Camera poses (SE3)
- d: Inverse depths
- δ_kj: Predicted flow corrections
- Σ_kj: Confidence weights (Mahalanobis distance)
```

**Two Modes:**
1. **Local BA**: Recent frames (sliding window of ~10 frames)
2. **Global BA**: All frames + inactive edges (for loop closure)

**Implementation:**
- Block-sparse Hessian structure
- Schur complement trick for speed
- 2 Gauss-Newton iterations
- Differentiable (backprop through optimization)

### 5. Loop Closure (`dpvo/loop_closure/`)

**DPV-SLAM extensions:**

#### Proximity Loop Closure (long_term.py)
- Uses camera proximity to detect loops
- Key innovation: **Uni-directional edges**
  - Saves memory (9.1GB → 0.6GB per 1K frames!)
  - Old patches → Recent frames (not bidirectional)
- Runs on GPU in main thread
- Fast: 0.1-0.18s per loop (vs 0.5-5s for DROID)

#### Classical Backend (DBoW2)
- Image retrieval using bag-of-words
- DISK keypoints + LightGlue matching
- Sim(3) alignment with RANSAC
- Pose graph optimization on CPU
- Corrects scale drift

## Data Flow Example

```
Frame t arrives
     ↓
1. Extract features (ResNet)
   - Matching: [1, 128, H/4, W/4]
   - Context: [1, 384, H/4, W/4]
     ↓
2. Sample M=96 patches randomly
   - Locations: [96, 3, 3, 3] (x, y, 1, depth)
   - Features: [96, 128, 3, 3] and [96, 384]
     ↓
3. Initialize pose (damped motion model)
   - T_new = exp(0.5 * log(T_{t-1} * T_{t-2}^{-1})) * T_{t-1}
     ↓
4. Add edges to patch graph
   - Forward: patches from t-13:t-1 → frame t
   - Backward: patches from t → frames t-13:t-1
   - Total: ~1200 new edges per frame
     ↓
5. Run update operator
   a) Reproject all patches
   b) Compute correlation volumes
   c) Network predicts delta and weight
   d) Bundle adjustment optimizes T and d
     ↓
6. Check keyframing
   - If motion < 15px between t-3 and t-1:
     Remove frame t-2
     Store relative pose for interpolation
     ↓
7. Loop closure (optional)
   - Every 10 frames: add proximity edges
   - DBoW2: retrieve similar images, PGO
     ↓
Output: Updated poses and 3D points
```

## Key Equations from Papers

### DPVO Paper

**Equation 1 - Patch Representation:**
```
P_k = [x, y, 1, d]ᵀ    where x,y,d ∈ ℝ^(1×p²)
```
Each patch is a 3×3 grid with constant inverse depth.

**Equation 2 - Reprojection:**
```
P'_kj ∼ K·T_j·T_i^(-1)·K^(-1)·P_k
```

**Equation 4 - Correlation:**
```
C(u,v,α,β) = ⟨g(u,v), f(P'(u,v) + Δ_αβ)⟩
```
Dot product between patch features and sampled frame features.

**Equation 6 - Bundle Adjustment:**
```
argmin  Σ ||Π[T_j^(-1)·T_i·Π^(-1)(P_k)] - I_kj||²_Σ
 T,d

Where I_kj = P'_kj + δ_kj (target after correction)
```

### DPV-SLAM Paper

**Uni-directional Edges (Fig. 2):**
- Only store features for destination frames
- Factors constrain both source and destination poses
- Massive memory savings for loop closure

**Classical Backend (Section 3.3):**
```
Smoothness term:  r_i = log(ΔS_{i,i+1}^(-1)·S_i^(-1)·S_{i+1})
Loop term:        r_jk = log(ΔS_jk^loop·S_j^(-1)·S_k)

Optimize: argmin Σ||r_i||² + Σ||r_jk||²
           S
```

## Performance Characteristics

### Speed (on RTX-3090):
- DPVO Default: 60 FPS average, 48 FPS worst-case
- DPVO Fast: 120 FPS average, 98 FPS worst-case
- DROID-SLAM: 40 FPS average, 11 FPS worst-case

### Memory:
- DPVO: 4.9GB GPU memory
- DPV-SLAM: 5-7GB GPU memory
- DROID-SLAM: 8.7GB GPU memory

### Accuracy (ATE in meters):
- EuRoC: 0.105 (DPVO) → 0.024 (DPV-SLAM) vs 0.022 (DROID)
- TartanAir: 0.21 (DPV-SLAM) vs 0.33 (DROID-SLAM)
- KITTI: 25.76m (DPV-SLAM++) vs 54.19m (DROID-SLAM)

## Configuration Parameters

From `config/default.yaml`:

```yaml
PATCHES_PER_FRAME: 96        # M patches per frame
BUFFER_SIZE: 2048            # Max keyframes
OPTIMIZATION_WINDOW: 10      # Local BA window
REMOVAL_WINDOW: 22           # Frames kept before removal
PATCH_LIFETIME: 13           # Frames to track each patch
KEYFRAME_THRESH: 15.0        # Motion threshold (pixels)
MOTION_MODEL: 'DAMPED_LINEAR'
MOTION_DAMPING: 0.5          # Damping coefficient
MIXED_PRECISION: True        # Use FP16
LOOP_CLOSURE: False          # Enable proximity LC
CLASSIC_LOOP_CLOSURE: False  # Enable DBoW2 backend
```

## Training

**Dataset:** TartanAir (synthetic, 50+ environments)

**Loss Function:**
```
L = 10·L_pose + 0.1·L_flow

L_pose = Σ ||log(G_i^(-1)·G_j)^(-1)·(T_i^(-1)·T_j)||
L_flow = min over p×p grid of ||flow_pred - flow_gt||
```

**Strategy:**
- 240k iterations, batch size 1
- 15-frame sequences
- First 1000 steps: structure-only (fix poses)
- Then: jointly optimize poses and depths

**Augmentation:**
- Random cropping
- Color jitter
- Random grayscale
- Color inversion

## File Structure Summary

```
dpvo/
├── dpvo.py              # Main DPVO class (★ HEAVILY COMMENTED)
├── net.py               # VONet architecture (★ COMMENTED)
├── patchgraph.py        # Patch graph data structure
├── ba.py                # Bundle adjustment interface
├── projective_ops.py    # Geometric transformations
├── extractor.py         # ResNet feature extractors
├── blocks.py            # Network building blocks
├── config.py            # Configuration management
│
├── altcorr/             # CUDA correlation kernel
│   ├── correlation.cpp
│   └── correlation.cu
│
├── fastba/              # CUDA bundle adjustment
│   ├── ba.cpp
│   ├── ba.cu
│   └── block_e.cu       # Block-sparse solver
│
├── lietorch/            # Lie group operations
│   └── src/*.cu         # SE3, Sim3, SO3
│
└── loop_closure/        # DPV-SLAM extensions
    ├── long_term.py     # Proximity LC
    └── retrieval/       # DBoW2 backend
```

## Key Insights

1. **Random Patch Sampling**: Surprisingly, random sampling works better than sophisticated keypoint detectors (ORB, SIFT, SuperPoint)!

2. **Sparse > Dense**: Tracking sparse patches is both faster AND more accurate than dense optical flow.

3. **Uni-directional Edges**: Brilliant memory optimization for loop closure - only store features for destination frames.

4. **Constant Runtime**: DPVO's frame rate is very stable (~60 FPS) unlike DROID which varies wildly (11-40 FPS).

5. **End-to-End Learning**: Supervising poses (not just flow) enables learning robust outlier rejection via confidence weights.

6. **Message Passing**: Graph neural network aggregation shares information between spatially separated patches.

## Usage Example

```python
from dpvo import DPVO

# Initialize
dpvo = DPVO(cfg, network='dpvo.pth', ht=480, wd=640, viz=True)

# Process video
for t, image in enumerate(video):
    intrinsics = [fx, fy, cx, cy]
    dpvo(t, image, intrinsics)

# Get results
poses, timestamps = dpvo.terminate()
```

## Citation

```bibtex
@inproceedings{teed2023dpvo,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  booktitle={NeurIPS},
  year={2023}
}

@article{lipson2024dpvslam,
  title={Deep Patch Visual SLAM},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  journal={arXiv preprint arXiv:2408.01654},
  year={2024}
}
```
