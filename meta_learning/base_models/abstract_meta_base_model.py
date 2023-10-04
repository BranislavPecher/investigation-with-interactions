import torch
import random
import numpy as np

from torchmeta.modules import MetaModule
from torch.autograd import Variable

class DeterministicModel():
    def __init__(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

    def set_rng_state(self, seed):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state

    def restore_rng_state(self, states):
        old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state = states

        torch.set_rng_state(old_torch_state)
        torch.cuda.set_rng_state(old_torch_cuda_state)
        np.random.set_state(old_numpy_state)
        random.setstate(old_random_state)

    def get_rng_state(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state


class AbstractMetaBaseModel(MetaModule, DeterministicModel):
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

