import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torcheval.metrics.aggregation.auc import AUC

class PixelScore():
    def __init__(self,model,input,heatmap,target,score):
        self.model=model
        self.input=input
        self.heatmap=heatmap
        self.target=target
        self.score=score
        self.result=None
    @staticmethod
    def __dilate(image):
        kernel = torch.ones((1, 1, 3, 3), device=image.device)
        result = F.conv2d(image[None, ...], kernel, padding=1)
        dilated_mask = (result > 0).float()[0, ...]
        return dilated_mask
    @staticmethod
    def __erode(image):
        kernel = torch.ones((1, 1, 3, 3), device=image.device)
        # manually padding with ones since pytorch only can pad with zeros in conv2d
        # padding = torch.ones((image.shape[0] + 2, image.shape[1] + 2), device=image.device)
        # padding[1:-1, 1:-1] = image
        padding = image
        batch = padding[None, ...] # [channels, height, width]
        result = F.conv2d(batch, kernel, padding=1)
        eroded_mask = (result == 9).float()[0, ...]
        return eroded_mask
    @staticmethod
    def __get_pixels(image):
        h,w=image.shape
        return torch.sum(image == 1).item() / (h*w)
    def run_erosion(self, target_fraction=0.01, threshold=0.5, MAX_ITE=100, callbacks=[]):
        heatmap = self.heatmap.squeeze().detach().clone()
        eroded_masks = (heatmap > threshold).float()
        self.result = [[(1.0,score)] for score in self.score.tolist()]
        for b, eroded_mask in enumerate(eroded_masks):
            ite = 0
            while ((n_pix := self.__get_pixels(eroded_mask)) > target_fraction) and (ite < MAX_ITE):
                eroded_mask = self.__erode(eroded_mask)
                with torch.no_grad():
                    output = self.model(self.input[b].unsqueeze(0) * eroded_mask)
                    score = output[0, self.target[b]].item()
                self.result[b].append((n_pix,score))
                for callback in callbacks:
                    callback(eroded_mask.unsqueeze(0))
                    #callback(ite,eroded_masks,eroded_mask)
                ite += 1
        return self.result
    def run_dilation(self, target_fraction=0.95, threshold=0.5, MAX_ITE=100, callbacks=[]):
        heatmap = self.heatmap.squeeze().detach().clone()
        dilated_masks = (heatmap > threshold).float()
        self.result = [[] for score in self.score.tolist()]
        for b, dilated_mask in enumerate(dilated_masks):
            ite = 0
            while ((n_pix := self.__get_pixels(dilated_mask)) < target_fraction) and (ite < MAX_ITE):
                dilated_mask = self.__dilate(dilated_mask)
                with torch.no_grad():
                    output = self.model(self.input[b].unsqueeze(0) * dilated_mask)
                    score = output[0, self.target[b]].item()
                self.result[b].append((n_pix,score))
                for callback in callbacks:
                    callback(dilated_mask.unsqueeze(0))
                    #callback(ite,dilated_masks,dilated_mask)
                ite += 1
            self.result[b].append((1.0,self.score[b].item()))
        return self.result
    def get_auc(self,):
        auc_results=[]
        for data in self.result:
            xy=torch.tensor(data)
            auc_score = AUC(reorder=True)
            scaler = MinMaxScaler()
            y_scaled=scaler.fit_transform(xy[:,1].unsqueeze(1)).squeeze() #norm for comparability
            y_torch = torch.from_numpy(y_scaled)
            auc_score.update(xy[:,0], y_torch)
            auc_results.append(auc_score.compute().item())
        auc_tensor = torch.tensor(auc_results)
        return auc_tensor
 