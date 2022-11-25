# pytorch-quaternion-geodesic-loss
pytorch quaternion geodesic loss

This criterion computes the angle difference(rad) between two quaternions. The equation is derived from [here](http://www.boris-belousov.net/2016/12/01/quat-dist/#using-quaternions).

```python
import torch
from geodesic_loss import GeodesicLoss

# batch_size of 2
pred = torch.tensor([[0.8734, -0.4822,  0.0327, -0.0595], [-0.8571,  0.3997, -0.2945, -0.1373]]) # predicted quatrenion
target = torch.tensor([[0.8710, -0.4866,  0.0452, -0.0498], [-0.8149,  0.0319, -0.3913, -0.4265]]) # ground truth quaternion

criterion = GeodesicLoss()
loss = criterion(pred, target)
print(loss.item(), 'rad')
```
