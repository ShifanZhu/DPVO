import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter


class LayerNorm1D(nn.Module):
    """
    Layer normalization for 1D sequences (e.g., graph nodes).

    Applies LayerNorm across the feature dimension by temporarily
    transposing to standard format expected by nn.LayerNorm.
    """
    def __init__(self, dim):
        """
        Args:
            dim: Feature dimension to normalize
        """
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, dim, length]

        Returns:
            Normalized tensor of same shape
        """
        # Transpose to [batch, length, dim], apply norm, transpose back
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    """
    Gated residual connection with learnable gating mechanism.

    Implements: output = x + gate(x) * residual(x)
    where gate learns to control information flow through the residual path.
    """
    def __init__(self, dim):
        """
        Args:
            dim: Feature dimension
        """
        super().__init__()

        # Gating mechanism outputs values in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        # Residual transformation
        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        """
        Args:
            x: Input features of shape [..., dim]

        Returns:
            Gated residual output of same shape
        """
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    """
    Soft attention-based aggregation for grouping features.

    Aggregates features belonging to the same group using learned attention weights.
    Implements: output = h(sum(f(x) * softmax(g(x)))) where grouping is by index ix.
    """
    def __init__(self, dim=512, expand=True):
        """
        Args:
            dim: Feature dimension
            expand: If True, expand aggregated features back to original size
        """
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        # Three linear transformations for value, attention, and output
        self.f = nn.Linear(self.dim, self.dim)  # Value transformation
        self.g = nn.Linear(self.dim, self.dim)  # Attention scoring
        self.h = nn.Linear(self.dim, self.dim)  # Output transformation

    def forward(self, x, ix):
        """
        Args:
            x: Input features of shape [batch, num_elements, dim]
            ix: Group indices indicating which elements belong together

        Returns:
            Aggregated features, either per-group or expanded back to original size
        """
        # Map indices to contiguous range [0, num_groups-1]
        _, jx = torch.unique(ix, return_inverse=True)

        # Compute attention weights within each group using softmax
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)

        # Weighted sum of transformed features within each group
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            # Expand per-group features back to original element positions
            return self.h(y)[:,jx]

        return self.h(y)

class SoftAggBasic(nn.Module):
    """
    Simplified soft aggregation with scalar attention weights.

    Similar to SoftAgg but uses a scalar attention score per element
    instead of a full feature-dimensional score, making it more efficient.
    """
    def __init__(self, dim=512, expand=True):
        """
        Args:
            dim: Feature dimension
            expand: If True, expand aggregated features back to original size
        """
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)  # Value transformation
        self.g = nn.Linear(self.dim, 1)          # Scalar attention scoring
        self.h = nn.Linear(self.dim, self.dim)  # Output transformation

    def forward(self, x, ix):
        """
        Args:
            x: Input features of shape [batch, num_elements, dim]
            ix: Group indices indicating which elements belong together

        Returns:
            Aggregated features, either per-group or expanded back to original size
        """
        # Map indices to contiguous range
        _, jx = torch.unique(ix, return_inverse=True)

        # Compute scalar attention weights within each group
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)

        # Weighted sum of transformed features
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            # Expand per-group features back to original positions
            return self.h(y)[:,jx]

        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1  # Global gradient clipping threshold

class GradClip(torch.autograd.Function):
    """
    Custom autograd function for gradient clipping.

    Forward pass is identity; backward pass clamps gradients to [-0.01, 0.01]
    and replaces NaN gradients with zeros.
    """
    @staticmethod
    def forward(ctx, x):
        """Identity forward pass."""
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Clip gradients to prevent exploding gradients.

        Args:
            grad_x: Gradient tensor

        Returns:
            Clipped gradient with NaNs replaced by zeros
        """
        # Replace NaN gradients with zeros
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        # Clamp to [-0.01, 0.01] range
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    """
    Module wrapper for gradient clipping operation.

    Use this as a layer in your network to clip gradients during backprop.
    """
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        """Apply gradient clipping via custom autograd function."""
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    """
    Custom autograd function that zeros out large gradients.

    Forward pass is identity; backward pass zeros gradients exceeding GRAD_CLIP threshold
    and replaces NaN gradients with zeros.
    """
    @staticmethod
    def forward(ctx, x):
        """Identity forward pass."""
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Zero out large gradients instead of clipping them.

        Args:
            grad_x: Gradient tensor

        Returns:
            Gradient with NaNs and large values set to zero
        """
        # Replace NaN gradients with zeros
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        # Zero out gradients larger than threshold (more aggressive than clipping)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    """
    Module wrapper for gradient zeroing operation.

    Use this to aggressively suppress large gradients by setting them to zero
    instead of clipping them.
    """
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        """Apply gradient zeroing via custom autograd function."""
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    """
    Debug utility to monitor gradient magnitudes.

    Forward pass is identity; backward pass prints mean absolute gradient
    value for debugging purposes.
    """
    @staticmethod
    def forward(ctx, x):
        """Identity forward pass."""
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Print gradient magnitude for debugging.

        Args:
            grad_x: Gradient tensor

        Returns:
            Unmodified gradient
        """
        print(grad_x.abs().mean())
        return grad_x
