import torch

class LinearOperator(torch.nn.Module):
    def __init__(self):
        super(LinearOperator, self).__init__()

    def forward(self, x):
        pass
        #在 forward 方法中，pass 是占位符，表示这个方法
        #目前没有具体实现。它用于定义一个接口或抽象类，供子类重写具体实现。

    def adjoint(self, x):
        pass

    def gramian(self, x):
        return self.adjoint(self.forward(x))

class SelfAdjointLinearOperator(LinearOperator):
    def adjoint(self, x):
        return self.forward(x)

class Identity(SelfAdjointLinearOperator):
    def forward(self, x):
        return x

class OperatorPlusNoise(torch.nn.Module):
    def __init__(self, operator, noise_sigma):
        super(OperatorPlusNoise, self).__init__()
        self.internal_operator = operator
        self.noise_sigma = noise_sigma

    def forward(self, x):
        A_x = self.internal_operator(x)
        return A_x + self.noise_sigma * torch.randn_like(A_x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#这是 operator 文件中的代码，主要定义了几个线性操作符类：

#LinearOperator: 基础类，定义了 forward、adjoint 和 gramian 方法，gramian 计算的是 
#𝐴^𝑇𝐴

#SelfAdjointLinearOperator: 自伴随线性操作符，adjoint 方法直接返回 forward 的结果，表示 
#𝐴=𝐴^𝑇

#Identity: 恒等操作符，forward 方法直接返回输入 𝑥

#OperatorPlusNoise: 接受一个操作符和噪声标准差 noise_sigma，在 forward 方法中，将操作符应用于输入 𝑥并添加噪声，表示 
#A(𝑥)+noise
#这些类用于定义线性变换和噪声处理，具体变换取决于传入的操作符实例。
#
#
#
#
#
#
#
#
#
#
#
#
#
#