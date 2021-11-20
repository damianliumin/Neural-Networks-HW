# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch
import torch.nn as nn

from model.ModelManager import load_checkpoint

class Ensemble(nn.Module):
    def __init__(self, model_list, args):
        """
        @param models (list): [model_name, ...]
        """
        super(Ensemble, self).__init__()
        self.name = args.model_name
        self.models = nn.ModuleList()
        for model_name in model_list:
            model, _ = load_checkpoint(f'{args.save_dir}/{model_name}_checkpoint_last.pt')
            self.models.append(model)

    def eval(self):
        for model in self.models:
            model.eval()    
        
    def forward(self, x):
        out = None
        for model in self.models:
            t = model(x).unsqueeze(1)  # b x 1 x 10
            if out is None:
                out = t
            else:
                out = torch.cat([out, t], 1)
        return out.mean(1).squeeze(1)



