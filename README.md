# High-fidelity-Multi-view-Normal-Integration-with-Scale-encoded-Neural-Surface-Representation
The code for "High-fidelity Multi-view Normal Integration with Scale-encoded Neural Surface Representation".

# Abstract 
Previous multi-view normal integration methods typically sample a single ray per pixel, without considering the spatial area covered by each pixel, which varies with camera intrinsics and the camera-to-object distance. Consequently, when the target object is captured at different distances, the normals at corresponding pixels may differ across views.
This multi-view surface normal inconsistency results in the blurring of high-frequency details in the reconstructed surface. To address this issue, we propose a scale-encoded neural surface representation that incorporates the pixel coverage area into the neural representation. By associating each 3D point with a spatial scale and calculating its normal from a hybrid grid-based encoding, our method effectively represents multi-scale surface normals captured at varying distances. Furthermore, to enable scale-aware surface reconstruction, we introduce a mesh extraction module that assigns an optimal local scale to each vertex based on the training observations. Experimental results demonstrate that our approach consistently yields high-fidelity surface reconstruction from normals observed at varying distances, outperforming existing multi-view normal integration methods.

# Install
**Simple version:** The installation procedure follows the same implementation as SuperNormal. Please refer to the link: [SuperNormal](https://github.com/CyberAgentAILab/SuperNormal)

**Complex version:**

```bash
conda create -n sn python=3.8
conda activate sn
pip install -r requirements.txt
```
Both work, but the requirements.txt file contains some unnecessary packages.

# Training
```bash
python exp_runner.py --case $object_name --conf exp1.conf
```

# Acknowledgements
Thanks to [NeuS](https://github.com/Totoro97/NeuS) and [SuperNormal](https://github.com/CyberAgentAILab/SuperNormal)!
