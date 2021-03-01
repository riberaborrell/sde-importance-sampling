import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, dim_in, dim_1, dim_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_1, bias=True)
        self.linear2 = nn.Linear(dim_1, dim_out, bias=True)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def reset_parameters(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def zero_parameters(self):
        for layer in self.children():
            for key in layer._parameters:
                layer._parameters[key] = torch.zeros_like(
                    layer._parameters[key], requires_grad=True
                )
