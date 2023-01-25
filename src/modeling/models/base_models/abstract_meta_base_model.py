import torch

from torchmeta.modules import MetaModule
from torch.autograd import Variable


class AbstractMetaBaseModel(MetaModule):
    def __init__(self) -> None:
        super(AbstractMetaBaseModel, self).__init__()
    
    def point_grad_to(self, target, device=None):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).to(device)
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

