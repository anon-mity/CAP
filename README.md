# CAP: Learning 3D Canonical Shape Representations Aligned with Human Preferences
We introduce Cap, a human-centric category-level point cloud canonicalization method that achieves SOTA consistency and aligns with human preferences. It exhibits improved physical significance, thereby driving numerous downstream tasks
![CAP](./img1_.png)

- [Project Page](https://anonymity15333.github.io/CAP.github.io/)
- [Dataset Page](https://anonymity15333.github.io/CAP.github.io/)
  
## Overview
![Overview](./img2.png)

## Usage
1.Download the ShapeNet dataset [here](https://condor-datasets.s3.us-east-2.amazonaws.com/dataset/ShapeNetAtlasNetH5_1024.zip) (Follow the data splitting of [ConDor](https://github.com/brown-ivl/ConDor) and [AtlasNet](https://github.com/TheoDEPRELLE/AtlasNetV2))
```python
# Create dataset directory
mkdir dataset
# Change directory
cd dataset
# Download the dataset (AtlasNet)
wget https://condor-datasets.s3.us-east-2.amazonaws.com/dataset/ShapeNetAtlasNetH5_1024.zip 
# Unzip the dataset
unzip ShapeNetAtlasNetH5_1024.zip
```

2.Training
```python
cd script
python train.py
```

## Data generation
1.Physics Simulator
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="./Cluster1.png" alt="Cluster1" width="400"/>
</div>

2.Cluster
<div style="text-align: center;">
  <img src="./Cluster4.png" alt="Cluster4" width="700"/>
</div>

## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed from this pytorch implementataion of [VNN](https://github.com/FlyingGiraffe/vnn) and [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE).
