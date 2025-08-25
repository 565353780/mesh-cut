import torch


def toVisiableVertexCurvature(curvatures: torch.Tensor) -> torch.Tensor:
    percentile_low, percentile_high = torch.quantile(
        curvatures, torch.tensor([0.02, 0.98], device=curvatures.device)
    )
    curvatures = torch.clamp(curvatures, percentile_low, percentile_high)

    scale_factor = 3.0
    curvatures = torch.tanh(scale_factor * curvatures)

    return curvatures
