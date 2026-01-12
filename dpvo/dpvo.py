"""
Deep Patch Visual Odometry (DPVO) - Main System

This file implements the core DPVO system from the papers:
- "Deep Patch Visual Odometry" (NeurIPS 2023)
- "Deep Patch Visual SLAM" (arXiv 2024)

Key Components:
1. Patch Graph: Sparse scene representation using 3x3 patches
2. Update Operator: Recurrent network for iterative refinement
3. Bundle Adjustment: Differentiable optimization layer
4. Loop Closure: Optional SLAM backend (proximity + classical)

The system processes video frames to estimate camera poses and 3D structure
by tracking sparse patches through time using deep learned features.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *

mp.set_start_method('spawn', True)

# Automatic mixed precision context manager for faster inference
autocast = torch.cuda.amp.autocast
# Identity transformation in SE(3) Lie group
Id = SE3.Identity(1, device="cuda")


class DPVO:
    """
    Main DPVO class - implements visual odometry/SLAM system

    Architecture:
    - Processes frames one at a time in __call__()
    - Maintains a sliding window of keyframes (buffer)
    - Tracks sparse 3x3 patches using correlation volumes
    - Iteratively refines poses and depths via update operator + BA
    - Optional loop closure for drift correction (SLAM mode)
    """

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        """
        Initialize DPVO system

        Args:
            cfg: Configuration object with hyperparameters
            network: VONet neural network (or path to checkpoint)
            ht: Input image height (default 480)
            wd: Input image width (default 640)
            viz: Enable real-time 3D visualization (default False)

        Key Parameters (from cfg):
            PATCHES_PER_FRAME: Number of patches per frame (default 96)
            BUFFER_SIZE: Max number of keyframes to keep (default 2048)
            OPTIMIZATION_WINDOW: Local BA window size (default 10)
            LOOP_CLOSURE: Enable proximity loop closure (default False)
            CLASSIC_LOOP_CLOSURE: Enable DBoW2 backend (default False)
        """
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False  # True after 8 frames processed
        self.enable_timing = False   # For profiling/debugging
        torch.set_num_threads(2)

        # M = patches per frame, N = buffer size (max keyframes)
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM  # Context feature dimension (384)
        RES = self.RES  # Resolution divisor (4)

        ### State Attributes ###
        self.tlist = []      # List of timestamps
        self.counter = 0     # Total frames processed (including non-keyframes)

        # Track global-BA calls to avoid redundant optimization
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        # Feature map resolution (1/4 of input due to network downsampling)
        ht = ht // RES
        wd = wd // RES

        # Dummy image for visualization updates
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### Network Attributes - Mixed Precision ###
        # Using half precision (FP16) significantly speeds up inference
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### Frame Memory Size ###
        # pmem: patch memory (how many frames worth of patches to store)
        # mem: frame memory (how many frames worth of features to store)
        self.pmem = self.mem = 36  # 32 was too small for default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000  # Track time since last global optimization
            self.pmem = self.cfg.MAX_EDGE_AGE  # Store more patches for loop closure

        # Context features for each patch (used by update operator)
        # Shape: [pmem, M, DIM] = [36, 96, 384]
        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)

        # Matching features for each patch (used for correlation)
        # Shape: [pmem, M, 128, P, P] = [36, 96, 128, 3, 3]
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        # Patch graph: stores poses, patches, edges, and factors
        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # Classical loop closure backend (DBoW2 + DISK features)
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        # Feature pyramid for correlation volumes
        # Level 1: 1/4 resolution, Level 2: 1/16 resolution
        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)
        self.pyramid = (self.fmap1_, self.fmap2_)

        # Optional real-time 3D viewer
        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """
        Compute correlation volumes for patch matching

        This implements Equation 4 from the DPVO paper:
        C(u,v,α,β) = ⟨g(u,v), f(P'(u,v) + Δ_αβ)⟩

        For each patch-frame edge (k,j):
        1. Sample a 7×7 grid around the reprojected patch center
        2. Compute dot products between patch features and frame features
        3. Creates a correlation volume showing visual similarity

        Args:
            coords: Reprojected patch coordinates [1, E, 3, 3, 2]
            indicies: Optional (ii, jj) edge indices

        Returns:
            Correlation features [1, E, 2*3*3*7*7] (2 pyramid levels)

        Note: Uses custom CUDA kernel (altcorr) for efficiency
        """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)  # Wrap around patch buffer
        jj1 = jj % (self.mem)            # Wrap around frame buffer

        # Level 1: 1/4 resolution (radius=3 → 7x7 grid)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        # Level 2: 1/16 resolution (radius=3 → 7x7 grid)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)

        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """
        Reproject patches from source frame to target frame

        Implements Equation 2 from the DPVO paper:
        P'_kj ∼ K·T_j·T_i^(-1)·K^(-1)·P_k

        Where:
        - P_k is patch k from frame i (3x3 grid with depth)
        - T_i, T_j are camera poses (SE3)
        - K is camera intrinsics
        - P'_kj is the reprojection into frame j

        Args:
            indicies: Optional (ii, jj, kk) tuple specifying:
                      ii = source frame indices
                      jj = destination frame indices
                      kk = patch indices

        Returns:
            Reprojected coordinates [1, E, 3, 3, 2] where E = num edges
        """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def update(self):
        """
        Core update step - runs one iteration of the update operator + BA

        This is the heart of DPVO, alternating between:
        1. Network Update: Predict flow corrections (delta) and weights
        2. Bundle Adjustment: Optimize poses and depths

        Corresponds to Section 3.1 of the DPVO paper.

        Flow:
        ┌─────────────┐
        │  Reproject  │  Project patches using current pose/depth estimates
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Correlation │  Compute visual similarities (Eq. 4)
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   Update    │  Neural network predicts:
        │  Operator   │  - delta: 2D flow corrections [E, 2]
        └──────┬──────┘  - weight: confidence weights [E, 2]
               │
        ┌──────▼──────┐
        │   Bundle    │  Optimize poses T and depths d to minimize:
        │ Adjustment  │  ||ω_ij(T,P_k) - [P'_kj + δ_kj]||²_Σ (Eq. 6)
        └─────────────┘

        The update operator uses:
        - Correlation features (visual alignment)
        - Context features (semantic information)
        - Hidden state (temporal consistency via GRU)
        """
        with Timer("other", enabled=self.enable_timing):
            # Step 1: Reproject all patches using current pose/depth estimates
            coords = self.reproject()

            with autocast(enabled=True):
                # Step 2: Compute correlation volumes (visual similarity)
                corr = self.corr(coords)

                # Step 3: Get context features for each patch
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]

                # Step 4: Run update operator (recurrent network)
                # Updates hidden state and predicts flow corrections
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")  # Damping factor for BA
            weight = weight.float()

            # Target = where patches should be after applying corrections
            target = coords[...,self.P//2,self.P//2] + delta.float()

        # Store factors for bundle adjustment
        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # Decide between global BA (all edges) vs local BA (recent edges)
                # Global BA runs when loop closure edges exist
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    # Local BA: optimize only recent frames (sliding window)
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            # Update 3D point cloud for visualization
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """
        Main entry point: Process a new video frame

        This is called once per input frame and handles:
        1. Feature extraction (patchify)
        2. Pose initialization (motion model)
        3. Edge creation (patch graph connectivity)
        4. Update operator + BA
        5. Keyframing (remove redundant frames)
        6. Loop closure (optional)

        Args:
            tstamp: Frame timestamp (integer or float)
            image: RGB image tensor [H, W, 3], uint8
            intrinsics: Camera intrinsics [fx, fy, cx, cy]

        System States:
        - Not initialized (n < 8): Accumulate frames, check for motion
        - Initializing (n == 8): Run 12 update iterations
        - Running (n > 8): Single update + keyframing per frame

        Frame Flow:
        ┌────────────┐
        │  Patchify  │ Extract features + randomly sample patches
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ Init Pose  │ Use damped linear motion model
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ Add Edges  │ Connect patches to nearby frames
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │   Update   │ Network + BA (see update() method)
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ Keyframe?  │ Remove frames with low motion
        └────────────┘
        """

        # Classical loop closure: add frame to DBoW2 database
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        # Check buffer overflow
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        # Update visualization
        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        # Normalize image to [-0.5, 0.5]
        image = 2 * (image[None,None] / 255.0) - 0.5

        # === STEP 1: Feature Extraction & Patch Sampling ===
        # See DPVO paper Section 3, Fig. 2
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                    return_color=True)
            # fmap: frame features [1, 128, H/4, W/4] - for correlation
            # gmap: patch features [M, 128, 3, 3] - matching features
            # imap: context features [M, 384] - for update operator
            # patches: initial patch locations [M, 3, 3, 3] (x, y, depth)
            # clr: patch colors [M, 3] - for visualization

        ### Update State Attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # Store patch colors (RGB -> BGR for visualization)
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        # Update indexing structures
        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        # === STEP 2: Pose Initialization ===
        # Use damped linear motion model: predict pose from previous motion
        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])  # Previous pose
                P2 = SE3(self.pg.poses_[self.n-2])  # Two frames ago

                # Handle varying camera framerates
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)  # Time ratio

                # Compute velocity in SE(3) tangent space, apply damping
                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data  # Extrapolate
                self.pg.poses_[self.n] = tvec_qvec
            else:
                # Constant velocity: just copy previous pose
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # === STEP 3: Depth Initialization ===
        # Initialize patch inverse depths
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            # Use median depth from recent patches (more stable)
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### Store Network Features ###
        # Use circular buffers (% operator wraps around)
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        # === Motion Check (Pre-initialization) ===
        # Ensure sufficient camera motion before starting optimization
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:  # Less than 2 pixels median flow
                # Skip this frame (not enough motion)
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        # Increment frame and patch counters
        self.n += 1
        self.m += self.M

        # === STEP 4: Loop Closure (DPV-SLAM) ===
        # Add proximity-based loop closure edges (Section 3.2 of DPV-SLAM paper)
        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # === STEP 5: Add Odometry Edges ===
        # Forward: recent patches → current frame
        # Backward: current patches → recent frames
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # === STEP 6: Optimization ===
        if self.n == 8 and not self.is_initialized:
            # Initialization: 8 frames accumulated, run 12 iterations
            self.is_initialized = True
            for itr in range(12):
                self.update()

        elif self.is_initialized:
            # Normal operation: single update + keyframing
            self.update()
            self.keyframe()

        # === STEP 7: Classical Loop Closure (Optional) ===
        # DBoW2-based global optimization (Section 3.3 of DPV-SLAM paper)
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
