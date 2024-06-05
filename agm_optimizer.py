import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import math


class SMD_compress(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0, weight_decay = 0, dampening=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening)
        super(SMD_compress, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_compress, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            
#             all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf
    #           (1+eps) norm potential function
                eps = 0.1
                update = (1+eps)* (torch.abs(p.data)**eps) * torch.sign(p.data) - group['lr'] * d_p
                p.data = (torch.abs(update/(1+eps))**(1/eps)) * torch.sign(update)

        return loss 
    

class SMD_qnorm(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0, weight_decay = 0, dampening=0, q =3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 2.0 <= q:
            raise ValueError("Invalid q_norm value: {}".format(q))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, q = q)
        super(SMD_qnorm, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_qnorm, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            
#             all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
    #           q norm potential function
                update = (group['q'])* (torch.abs(p.data)**(group['q']-1)) * torch.sign(p.data) - group['lr'] * d_p
                p.data = (torch.abs(update/(group['q']))**(1/(group['q'] - 1))) * torch.sign(update)

        return loss
    
class AGM(Optimizer):

    def __init__(self, params, lr_tau=0.001, lr_eta=0.001, lr_alpha=0.001, momentum=0, weight_decay = 0, dampening=0, q =3):
        if not 0.0 <= lr_tau:
            raise ValueError("Invalid learning rate Tau: {}".format(lr_tau))
        if not 0.0 <= lr_eta:
            raise ValueError("Invalid learning rate Eta: {}".format(lr_eta))
        if not 0.0 <= lr_alpha:
            raise ValueError("Invalid learning rate Alpha: {}".format(lr_alpha))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 2.0 <= q:
            raise ValueError("Invalid q_norm value: {}".format(q))

        defaults = dict(lr_tau=lr_tau, lr_eta=lr_eta, lr_alpha=lr_alpha, momentum=momentum, weight_decay=weight_decay, dampening=dampening, q = q)
        super(AGM, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_qnorm, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            
#             all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
    #           q norm potential function
                y_p = p.data - group['lr_eta'] * d_p
                update = (group['q'])* (torch.abs(p.data)**(group['q']-1)) * torch.sign(p.data) - group['lr_alpha'] * d_p
                z_p = (torch.abs(update/(group['q']))**(1/(group['q'] - 1))) * torch.sign(update)
                p.data = group['lr_tau'] * y_p + (1 - group['lr_tau']) * z_p



        return loss
    