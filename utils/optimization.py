import torch
import torch.nn.functional as F
import time

class Optimizer:
  def __init__(self, optimizer, trainloader, device, update_centroids=False):
    self.optimizer = optimizer
    self.trainloader = trainloader
    self.n = len(trainloader.dataset)
    self.update_centroids = update_centroids
    self.device=device
    self.best_acc=0
    
  def gradient_penalty(self, inputs, outputs):
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

  def train_epoch(self, net, criterion, adversary=None, verbose=False):
    train_loss, correct, conf = 0, 0, 0
    start_time=time.time()
    net.train() 
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      if adversary is not None: # not feasible for SNGP because the covariance matrix is computed after each epoch.
        # when performing attack, the model needs to be in eval mode
        # also the parameters should NOT be accumulating gradients
        #with torch.no_grad(): #requires_grad=False & eval()
        net.eval()
        net.requires_grad_(False)
        criterion.requires_grad_(False)
        inputs = adversary.perturb(inputs, targets)
        net.requires_grad_(True)
        criterion.requires_grad_(True)
        net.train()
      #if weight_gp_pred + weight_gp_embed>0:
      #  inputs.requires_grad_(True)
      self.optimizer.zero_grad()
      #embedding = net.embed(inputs)
      loss, Y_pred = criterion.loss(inputs,targets, net)
      if verbose:
        print("loss:",loss.item())
      loss.backward()
      self.optimizer.step()
      inputs.requires_grad_(False)

      with torch.no_grad():
        criterion.prox(net)
        train_loss += loss.item()
        confBatch, predicted = Y_pred.max(1)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    execution_time = (time.time() - start_time)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f | time (s): %.2f'% (train_loss/len(self.trainloader), 100.*correct/self.n, correct, self.n, 100*conf/self.n, execution_time))
    return (100.*correct/self.n, 100*conf/self.n)
  
  def test_acc(self, net, criterion, data_loader, min_conf=0):
    net.eval()
    test_loss, correct, conf, total = 0,0,0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #outputs = net.embed(inputs)
            loss,Y_pred = criterion.loss(inputs, targets, net)

            test_loss += loss.item()
            confBatch, predicted = Y_pred.max(1)
            idx = (confBatch>min_conf)
            correct += predicted[idx].eq(targets[idx]).sum().item()
            conf+=confBatch[idx].sum().item()
            total+= idx.sum()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (test_loss/max(len(data_loader),1), 100.*correct/total, correct, total, 100*conf/total))
    return (100.*correct/total, 100*conf/total)
  
  def test_grad_penalty(self, net, criterion, data_loader, gp_embed):
    net.eval()
    gp = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs.requires_grad_(True)
        embedding = net.embed(inputs)
        criterion(embedding, targets)

        if not gp_embed:
          gp += self.gradient_penalty(inputs, criterion.Y_pred).item()
        if gp_embed:
          gp += self.gradient_penalty(inputs, embedding).item()
        inputs.requires_grad_(False)
    print('Gradient Penalty: %.3f'% (gp/max(len(data_loader),1)))
    return gp


  def optimize_centroids(self, net):
    net.eval()
    d,c = net.classifier.in_features,net.classifier.out_features
    Z=torch.zeros(d,c).to(self.device)
    y_sum = torch.zeros(c).to(self.device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            D = net.embed(inputs)
            Y = F.one_hot(targets, c).float().to(self.device)
            Z += D.t().mm(Y)
            y_sum += torch.sum(Y,0)
    Z = Z/y_sum
    net.classifier.weight.data = Z.t()
