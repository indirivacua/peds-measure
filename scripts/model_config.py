import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam, Occlusion
from .rise import RISE

class SaveActivations:
    def __init__(self):
        self.activations = None
    def __call__(self, module, module_in, module_out):
        self.activations = module_out

class ModelConfig():
    def __init__(self, model, input):
        self.model = model
        self.input = input
        self.model_preds = None
        self.target = None
        self.score = None
    def __call__(self,input):
        return self.__wrapped_model(input)
    def __wrapped_model(self,input):
        output = self.model(input)
        output = F.softmax(output, dim=1)
        return output
    def __get_model_layer(self,attr_str):
        attr_name, layer_idx = attr_str.split('[')
        layer_idx = int(layer_idx.split(']')[0])
        attr = getattr(self.model, attr_name)
        return attr, layer_idx
    def set_model_config(self, mode='eval', dtype=torch.float32, device=torch.device('cpu')):
        self.model = self.model.to(dtype).to(device)
        self.input = self.input.to(dtype).to(device)
        self.model.eval() if mode == 'eval' else self.model.train()
    def run_forward_pass(self,idx_to_labels,verbose=True):
        print(f"============ {self.model.__class__.__name__} ============") if verbose else None
        self.model_preds = []
        output = self.__wrapped_model(self.input)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        for score, label_idx in zip(prediction_score, pred_label_idx):
            id = label_idx.item()
            predicted_label = idx_to_labels[str(id)][1]
            print(f"Predicted: {predicted_label} ({id}) ({score.squeeze().item()})") if verbose else None
            self.model_preds.append({'id':id,'predicted_label':predicted_label,'score':score.squeeze().item()})
        self.target = pred_label_idx[:,0]
        self.score = prediction_score[:,0]
        return self.model_preds
    @staticmethod
    def heatmap_normalize(t):
        t = t.clone()
        # Iterate over each image in the batch
        for i in range(t.shape[0]):
            t[i] -= t[i].min()
            t[i] /= t[i].max()
        return t
    @staticmethod
    def heatmap_upsample(heatmap,upsample_shape:tuple[int,int]):
        heatmap = F.interpolate(
            heatmap,
            size=upsample_shape,
            mode="bicubic",#"bilinear",
            align_corners=True,
        ).squeeze()
        return heatmap
    def get_activations(self, attr_str, pool=False): #ex: "features[30]" / "layer4[1]"
        attr, layer_idx = self.__get_model_layer(attr_str)
        save_activations = SaveActivations()
        handle = attr[layer_idx].register_forward_hook(save_activations)
        self.model(self.input)
        handle.remove()
        if pool:
            save_activations.activations = torch.mean(save_activations.activations, dim=1, keepdim=True)
        return save_activations.activations
    def get_grad_cam(self, attr_str):
        attr, layer_idx = self.__get_model_layer(attr_str)
        layer_gc = LayerGradCam(self.model, attr[layer_idx])
        attributions_lgc = layer_gc.attribute(self.input, self.target)
        return attributions_lgc
    def get_occlusion(self, strides=(3, 8, 8),sliding_window_shapes=(3,15, 15),verbose=True):
        occlusion = Occlusion(self.model)
        attributions_occ = occlusion.attribute(self.input,
                                               strides=strides,
                                               target=self.target,
                                               sliding_window_shapes=sliding_window_shapes,
                                               baselines=0,
                                               show_progress=verbose)
        return attributions_occ
    def get_rise(self,n_masks, initial_mask_shape, verbose=True):
        rise = RISE(self.model)
        heatmap = rise.attribute(self.input,
                                n_masks=n_masks,
                                initial_mask_shapes=(initial_mask_shape,),
                                patience=n_masks,
                                target=self.target, show_progress=verbose)
        return heatmap
    def get_cbrise(self,n_masks, initial_mask_shape, blur_sigma=None, patience=128, d_epsilon=1e-2, threshold=0.1, verbose=True):
        rise = RISE(self.model)
        metrics=[]
        heatmaps=[]
        for input, target in zip(self.input, self.target):
            input = input.unsqueeze(dim=0)
            metric={}
            heatmap = rise.attribute(input,
                                    n_masks=n_masks,
                                    initial_mask_shapes=(initial_mask_shape,),
                                    blur_sigma=blur_sigma,
                                    target=target,
                                    patience=patience,
                                    d_epsilon=d_epsilon,
                                    threshold=threshold,
                                    show_progress=verbose,
                                    metrics=metric,
                                    )
            metrics.append(metric)
            heatmaps.append(heatmap)
        heatmaps = torch.cat(heatmaps,dim=0)
        return heatmaps, metrics
        