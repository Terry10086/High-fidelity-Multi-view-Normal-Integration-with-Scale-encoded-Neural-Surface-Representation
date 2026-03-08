import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import tinycudann as tcnn

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 real_data,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 bias=0.5,
                 radius_min = 0.0001,
                 radius_max = 0.0018,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 encoding_config=None,
                 input_concat=False):
        super(SDFNetwork, self).__init__()
        self.input_concat = input_concat
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.real_data = real_data
        
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if encoding_config is not None:
            self.encoding = tcnn.Encoding(d_in, encoding_config).to(torch.float32)
            dims[0] = self.encoding.n_output_dims
            if input_concat:
                dims[0] += d_in
        else:
            self.encoding = None

        dims[0] = 43   # 79  31  59

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.bindwidth = 0
        self.enc_dim = self.encoding.n_output_dims

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif self.encoding is not None and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.encoding is not None and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def increase_bandwidth(self):
        self.bindwidth += 1

    def forward(self, inputs, encoding, radius, set_radius=None, cal_sdf_loss=False, num_points_to_sample = 1, radius_sample = 1):  # num_points_to_sample = 4096, radius_sample = 128
        # torch.set_printoptions(precision=6)
        # print(radius.min(),radius.max())
        
        radius_ct = self.contraction(radius, self.radius_min, self.radius_max)
        radius_ct = torch.round(radius_ct * 100) / 100
        
        if encoding is not None:
            encoded = encoding(inputs, radius_ct)      # torch.Size([262144, 192])

            if cal_sdf_loss==True:
                set_radius = self.contraction(torch.tensor(set_radius), self.radius_min, self.radius_max)
                indices = torch.randperm(inputs.size(0))[:num_points_to_sample]
                inputs2 = inputs[indices].repeat_interleave(radius_sample, dim=0)
                
                # for sdf loss
                radius2 = torch.linspace(-1.0, 1.0, radius_sample).to(device = inputs.device).unsqueeze(-1).repeat(num_points_to_sample, 1)+ 0.0078 * torch.randn(num_points_to_sample*radius_sample).unsqueeze(-1)
                encoded2 = encoding(inputs2, radius2)

                encoded = torch.cat([encoded, encoded2], dim=0)
                inputs = torch.cat([inputs, inputs2], dim=0)        # sdf, radius_loss, mesh
            

                
        encoded_hash = self.encoding(inputs).to(torch.float32)
        sep = encoded.shape[1]
        sep_hash = encoded_hash.shape[1]

        if self.input_concat:
            inputs = torch.cat([inputs, encoded_hash, encoded], dim=1) 

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        
        
        if cal_sdf_loss == True:
            gap = radius_ct.shape[0]
            return x[:gap,:], x[gap:gap+num_points_to_sample*radius_sample,:].view(num_points_to_sample, radius_sample)         
        else:
            return x
        

    def sdf(self, x, encoding, level):
        return self.forward(x, encoding, level)
        

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    @torch.enable_grad()
    def gradient(self, x, encoding, level):
        x.requires_grad_(True)
        y = self.sdf(x, encoding, level)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def contraction(self, data, estimated_min, estimated_max):
        # 0~15.5
        data_normalized = (data - estimated_min) / (estimated_max - estimated_min)
    
        # 将数据缩放到 [-1, 1] 之间
        data_normalized = data_normalized * 2 - 1
        return data_normalized





class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)