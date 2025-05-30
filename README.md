# Mesh Graph Cut

基于图割算法的三角网格划分工具，可将三角网格划分为N个连续三角面片的集合，同时约束每个集合中的顶点（不包含边界）曲率之和近似相等，并且将高曲率顶点尽可能放置在边界上。

## 算法原理

该算法基于谱聚类和图割理论，主要步骤如下：

1. **曲率计算**：使用离散微分几何方法计算网格上每个顶点的平均曲率。
2. **构建三角形邻接图**：将三角网格表示为图，其中节点是三角形，边表示相邻三角形之间的连接。
3. **构建拉普拉斯矩阵**：基于曲率信息构建加权拉普拉斯矩阵，使得曲率差异大的三角形之间的权重较小。
4. **谱聚类**：计算拉普拉斯矩阵的特征向量，并使用K-means聚类将三角形分为N个子集。
5. **边界优化**：调整分割边界，使高曲率顶点尽可能位于边界上，从而最小化每个子网格内部的曲率之和。

## 安装

### 依赖项

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```python
from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter

# 加载网格
mesh_file_path = "path/to/your/mesh.obj"
mesh_graph_cutter = MeshGraphCutter(mesh_file_path)

# 将网格切割为10个子网格
mesh_graph_cutter.cutMesh(10)

# 可视化结果
mesh_graph_cutter.visualizeSegments()

# 导出分割结果
mesh_graph_cutter.exportSegments("output/segments")
```

### 完整示例

运行演示脚本：

```bash
python demo.py
```

## 功能特点

- **曲率感知分割**：考虑网格的曲率信息进行分割，使高曲率区域位于分割边界上
- **均衡分割**：约束每个子网格内部顶点的曲率之和近似相等
- **可视化工具**：提供网格曲率、分割结果和边界顶点的可视化功能
- **结果导出**：支持将分割结果导出为多个OBJ文件
- **分割评估**：提供分割质量的评估指标

## 算法参数调整

在`MeshGraphCutter`类中，可以调整以下参数以适应不同的网格和分割需求：

- `_buildLaplacianWithCurvature`方法中的权重系数（默认为5.0）
- `_optimizeBoundaries`方法中的迭代次数（默认为100）
- `_optimizeBoundaries`方法中考虑的高曲率顶点比例（默认为10%）

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。