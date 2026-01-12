import numpy as np
import torch
from einops import asnumpy, reduce, repeat

from . import projective_ops as pops
from .lietorch import SE3
from .loop_closure.optim_utils import reduce_edges
from .utils import *


class PatchGraph:
    """
    PatchGraph manages the spatial-temporal graph structure for DPVO.

    This class maintains a sliding window of camera poses, image patches, and their
    3D point correspondences. It also manages edges connecting patches across frames
    for bundle adjustment and loop closure detection.
    """

    def __init__(self, cfg, P, DIM, pmem, **kwargs):
        """
        Initialize the PatchGraph data structure.

        Args:
            cfg: Configuration object containing SLAM parameters
            P: Patch size (e.g., 3 for 3x3 patches)
            DIM: Feature dimension for network embeddings
            pmem: Patch memory size
            **kwargs: Additional arguments passed to tensor initialization
        """
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames currently in the graph
        self.m = 0      # number of patches currently in the graph

        # Maximum patches per frame and buffer size
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        # Frame metadata: timestamps for each frame
        self.tstamps_ = np.zeros(self.N, dtype=np.int64)

        # Camera poses stored as SE3 in quaternion format [tx, ty, tz, qx, qy, qz, qw]
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")

        # Image patches: [num_frames, patches_per_frame, 3(x,y,depth), patch_height, patch_width]
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")

        # Camera intrinsics: [fx, fy, cx, cy] for each frame
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        # 3D points reconstructed from patches
        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")

        # RGB colors for visualization of patches
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        # Frame index for each patch (which frame does each patch belong to)
        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")

        # Mapping from frame to its position in buffer
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # Initialize poses to identity transformation (quaternion [0,0,0,1])
        self.poses_[:,6] = 1.0

        # Store relative poses for frames removed from sliding window
        self.delta = {}

        ### Edge information for active correspondences ###
        # Network features for edges
        self.net = torch.zeros(1, 0, DIM, **kwargs)
        # Edge indices: ii = source frame, jj = target frame, kk = patch index
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        ### Inactive edge information (no longer updated, but retained for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        # Correlation weights for inactive edges
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        # Target coordinates for inactive edges
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")

    def edges_loop(self):
        """
        Create loop closure edges by connecting old patches to recent frames.

        This method identifies potential loop closures by creating edges between
        patches from older frames (outside the removal window) and recent frames.
        It filters edges based on optical flow magnitude to ensure geometric consistency.

        Returns:
            Tuple of (kk, jj) tensors representing patch indices and target frame indices
            for valid loop closure edges.
        """
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n - self.cfg.REMOVAL_WINDOW  # Upper bound for "old" patches

        if l <= 0:
            return torch.empty(2, 0, dtype=torch.long, device='cuda')

        # Create candidate edges between recent frames and old patches
        # jj: target frames (recent), kk: source patches (old)
        jj, kk = flatmeshgrid(
            torch.arange(self.n - self.cfg.GLOBAL_OPT_FREQ, self.n - self.cfg.KEYFRAME_INDEX, device="cuda"),
            torch.arange(max(l - lc_range, 0) * self.M, l * self.M, device="cuda"), indexing='ij')
        ii = self.ix[kk]  # Get frame indices for the patches

        # Compute optical flow magnitude to filter geometrically inconsistent edges
        flow_mg, val = pops.flow_mag(SE3(self.poses), self.patches[...,1,1].view(1,-1,3,1,1), self.intrinsics, ii, jj, kk, beta=0.5)

        # Aggregate flow magnitude across patches
        flow_mg_sum = reduce(flow_mg * val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).float()
        num_val = reduce(val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).clamp(min=1)

        # Compute average flow, requiring at least 75% valid patches
        flow_mag = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)

        # Filter edges with flow magnitude below threshold
        mask = (flow_mag < self.cfg.BACKEND_THRESH)

        # Reduce number of edges using non-maximum suppression
        es = reduce_edges(asnumpy(flow_mag[mask]), asnumpy(ii[::self.M][mask]), asnumpy(jj[::self.M][mask]), max_num_edges=1000, nms=1)

        # Expand frame-level edges to patch-level edges
        edges = torch.as_tensor(es, device=ii.device)
        ii, jj = repeat(edges, 'E ij -> ij E M', M=self.M, ij=2)
        kk = ii.mul(self.M) + torch.arange(self.M, device=ii.device)
        return kk.flatten(), jj.flatten()

    def normalize(self):
        """
        Normalize depth scale and update all related quantities.

        This method fixes the scale ambiguity inherent in monocular SLAM by:
        1. Computing the mean depth across all patches
        2. Normalizing depths to unit scale
        3. Scaling camera translations accordingly
        4. Re-centering poses relative to the first frame
        5. Recomputing 3D point cloud from normalized poses and depths
        """
        # Compute mean depth across all active patches
        s = self.patches_[:self.n,:,2].mean()

        # Normalize depth by scale factor
        self.patches_[:self.n,:,2] /= s

        # Scale translation component of poses (first 3 elements)
        self.poses_[:self.n,:3] *= s

        # Update relative poses stored in delta dictionary
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))

        # Re-center all poses relative to the first frame (make first pose identity)
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[[0]]).inv()).data

        # Reconstruct 3D point cloud from normalized poses and depths
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        # Convert from homogeneous to Euclidean coordinates (divide by w)
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

    @property
    def poses(self):
        """Return poses reshaped for batch processing [1, N, 7]."""
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        """Return patches reshaped to [1, N*M, 3, 3, 3] for projective operations."""
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        """Return intrinsics reshaped for batch processing [1, N, 4]."""
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        """Return flattened patch indices."""
        return self.index_.view(-1)
