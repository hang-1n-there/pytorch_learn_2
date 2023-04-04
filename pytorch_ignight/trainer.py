from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np

from ignite.engine import Engine
from ignite.engine import Events

from utils import get_parameter_norm
from utils import get_grad_norm

class MyEngine(Engine):
    def __init__(self,func,model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config
        
        super().__init__(func)
        
        self.best_loss = np.inf
        self.best_model = None
        
        self.device = next(model.parameters()).device
        
    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()
        
        x,y = mini_batch
        x,y = x.to(engine.device), y.to(engine.device)
        
        y_hat = engine.model(x)
        loss = engine.crit(y_hat,y)
        loss.backward()
        
        # isinstance = 인스턴스와 타입이 같으면 true를 반환
        # argmax = 차원에 따라 가장 큰 값의 인덱스를 반환
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        
        else :
            accuracy = 0
        
        # p_norm(parameter의 L2_nomalization)은 학습이 잘 진행되고 있는지와 비례한다
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        
        # gradient의 크기 , 학습의 정도
        g_norm = float(get_grad_norm(engine.model.parameters()))
        
        engine.optimizer.step()
        
        return { 'loss' : float(loss) , 'accuracy' : float(accuracy) , '|param|' : p_norm, '|g_param|' : g_norm}
    
    @staticmethod
    def validate(engine, mini_batch):
        
        with torch.no_grad:
            x , y = mini_batch
            x , y = x.to(engine.device) , y.to(engine.device)
            
            y_hat = engine.model(x)
            loss = engine.crit(y_hat, y)
            
            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y, dim=-1) == y).sum() / float(y.size(0))
            else :
                accuracy = 0
            
        return {'loss' : float(loss) , 'accuracy' : float(accuracy)}
    
class Trainer():
    def __init__(self,config):
        self.config = config
    
    def train(self, model, crit, optimizer, train_loader, valid_loader):
        
        train_engine = MyEngine(MyEngine.train, model, crit, optimizer, self.config)
        validation_engine = MyEngine(MyEngine.validate, model, crit, optimizer, self.config)
        
        
        