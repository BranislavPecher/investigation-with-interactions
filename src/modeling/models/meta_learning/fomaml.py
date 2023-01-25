from src.modeling.models.meta_learning.maml import MAML
import torch.nn.functional as F


class FOMAML(MAML):
    def __init__(self, model, meta_optimizer=None, base_optimizer=None, meta_lr=1e-3, base_lr=1e-3,
                 learn_step_size=False, per_param_step_size=False,
                 inner_iterations=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, meta_optimizer=meta_optimizer, first_order=True,
                                     base_lr=base_lr, learn_step_size=learn_step_size,
                                     per_param_step_size=per_param_step_size,
                                     inner_iterations=inner_iterations, scheduler=scheduler,
                                     loss_function=loss_function, device=device)
