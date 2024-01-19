import torch.nn as nn
import torch
import torch.nn.functional as F


def gradient_penalty(inputs, outputs):
    gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
        )[0]
    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty
    
class CE_Loss(nn.Module):
    def __init__(self, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        #self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        #self.logits = 0
 
    def forward(self, logits, targets):  
        #self.logits = self.classifier(inputs) # prediction before softmax
        return self.ce_loss(logits, targets)

    def loss(self, inputs, targets, net):
        logits = net(inputs)
        return self.ce_loss(logits, targets), self.conf_logits(logits,net)

    def conf_logits(self,logits, net):
        if hasattr(net.classifier,'conf_logits'):
            return net.classifier.conf_logits(logits)
        return self.softmax(logits)
        
    def conf(self, inputs, net):
        logits = net(inputs)
        return self.softmax(logits)
    
    def prox(self, net):
        return

