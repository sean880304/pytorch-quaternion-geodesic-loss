import torch
import torch.nn as nn
import torch.nn.functional as F

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()
        
    def forward(self, pred, target):
        # target = x, y, z, w, is the annotation
        pred_norm = F.normalize(pred, p=2, dim=1) # pred quaternion needs to be normalized
        inner_of_real_part = target[:,0]*pred_norm[:,0] + target[:,1]*pred_norm[:,1] + target[:,2]*pred_norm[:,2] + target[:,3]*pred_norm[:,3]
        theta = torch.acos(2 * (inner_of_real_part**2) - 1)
        # print(theta*180/np.pi)
        # print(pred_norm)
        # print(target)
        loss = theta.sum() / target.shape[0]
        
        return loss
