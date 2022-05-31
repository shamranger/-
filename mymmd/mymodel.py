import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,layers):
        super(VGG, self).__init__()
        self.net = nn.Sequential(*layers)
        self.outputs = list()
        self.net[0].register_forward_hook(self._hook)
        #print(self.net[0].register_forward_hook(self._hook))
        self.net[5].register_forward_hook(self._hook)
        #print(self.outputs)
        self.net[10].register_forward_hook(self._hook)
        #print(self.outputs)
        self.net[19].register_forward_hook(self._hook)
        self.net[28].register_forward_hook(self._hook)

    def _hook(self,module,input,output):
        self.outputs.append(output)
        #print(self.outputs)

    def forward(self,x):
        _ = self.net(x)
        #print(_)
        out = self.outputs.copy()
        #print(out)
        self.outputs = list()
        return out

