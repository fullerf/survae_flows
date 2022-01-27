import torch
import torch.nn as nn
import math


class HollowConv2d(torch.nn.Conv2d):
    """
        A Conv2d layer masked to so that the ouput does not depend upon the 
        value of the input at the center of the sliding filter.
        
        E.g. for a 3x3 kernel, the mask is:
        [[1 1 1],    
         [1 0 1],    
         [1 1 1]]
         
        For a 2x2 kernel, the mask is:
        [[0 1 ],    
         [1 1 ]]
    """

    def __init__(self, *args, **kwargs):
        """
        *args: as for nn.Conv2d
        **kwargs: as for nn.Conv2d
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.ones_like(self.weight)
        k = h // 2 if h % 2 > 0 else h // 2 - 1
        j = w // 2 if w % 2 > 0 else w // 2 - 1
        mask.data[:, :, k , j] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class HollowConv1d(torch.nn.Conv1d):
    """ 
        A Conv1d layer masked to so that the ouput does not depend upon the 
        value of the input at the center of the sliding filter.
        
        E.g. for a length 3 kernel, the mask is:
        [1 0 1]
         
        For a length 2 kernel, the mask is:
        [0 1 ]
    """

    def __init__(self, *args, **kwargs):
        """
        *args: as for nn.Conv1d
        **kwargs: as for nn.Conv1d
        """
        super().__init__(*args, **kwargs)
        i, o, w = self.weight.shape
        mask = torch.ones_like(self.weight)
        j = w // 2 if w % 2 > 0 else w // 2 - 1
        mask.data[:, :, j] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class HollowLinear(torch.nn.Linear):
    """
        A Linear layer masked so that rows of the output depend on all elements of
        the input save one, which is accomplished by applying a mask to the
        linear transform:
        
        E.g. the mask for input_dim = 3 and output_dim = 5, is:
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [0, 1, 1],
         [1, 0, 1]]
         
    """
    def __init__(self, *args, **kwargs):
        """
         Args:
            inputs: integer number of input dimensions
            outputs_per_input: integer number of output dimensions PER input dimension
            
        Kwargs:
            as for torch.nn.Linear
        """
        super().__init__(*args, **kwargs)
        o, i = self.weight.shape
        m = math.ceil(o/i)
        with torch.no_grad():
            mask = torch.logical_not(torch.diag_embed(torch.ones(m,i)).type(torch.bool))
            mask = mask.reshape(m*i, i).type(self.weight.dtype).to(self.weight.device)
            mask = mask[:o,:]
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# +
class SplitHollowLinear(torch.nn.Module):
    
    def __init__(self, outputs_per_input: int):
        super().__init__()
        self.outputs_per_input = outputs_per_input
        
    def forward(self, x):
        return x.split(x.shape[-1]//self.outputs_per_input, dim=-1)
    
class SplitHollowConv2d(torch.nn.Module):
    
    def __init__(self, outputs_per_input: int):
        super().__init__()
        self.outputs_per_input = outputs_per_input
        
    def forward(self, x):
        return x.split(x.shape[-3]//self.outputs_per_input, dim=-3)
    
class SplitHollowConv1d(torch.nn.Module):
    
    def __init__(self, outputs_per_input: int):
        super().__init__()
        self.outputs_per_input = outputs_per_input
        
    def forward(self, x):
        return x.split(x.shape[-2]//self.outputs_per_input, dim=-2)
