import numpy as np


def toVisiableVertexCurvature(curvatures: np.ndarray) -> np.ndarray:
    # 应用一些处理以增强曲率差异
    # 1. 去除异常值
    percentile_low, percentile_high = np.percentile(curvatures, [2, 98])
    curvatures = np.clip(curvatures, percentile_low, percentile_high)

    # 2. 应用非线性变换增强对比度
    # 使用双曲正切函数进行非线性变换，保留符号但增强对比度
    scale_factor = 3.0  # 控制变换的强度
    curvatures = np.tanh(scale_factor * curvatures)

    return curvatures
