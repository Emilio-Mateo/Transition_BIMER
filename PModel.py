import torch.nn as nn


class ConvClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding= 1) , #64
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8,3, padding= 1) , #32
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,3,padding= 1) , #16
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(2), # 8
            nn.Flatten() , # (512, 1)
            nn.Linear(512,4)
        )
    
    def forward(self,x):
        decoded = self.encoder(x)
        return decoded



class ConvClass2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding= 1) , #64
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8,3, padding= 1) , #32
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,3,padding= 1) , #16
            nn.SiLU(),
            nn.MaxPool2d(2), # 8
            nn.Flatten() , # (512, 1)
            nn.Linear(512,4)
        )
    
    def forward(self,x):
        decoded = self.encoder(x)
        return decoded


class ConvClass3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding= 1) , #64
            nn.LeakyReLU(0.01),
            nn.Dropout2d(p = 0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8,3, padding= 1) , #32
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,3,padding= 1) , #16
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2), # 8
            nn.Flatten() , # (512, 1)
            nn.Dropout(p= 0.3),
            nn.Linear(512,4),
        )
    
    def forward(self,x):
        decoded = self.encoder(x)
        return decoded


class LinearClass(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8,64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0,3),
            nn.Linear(64,32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0,3),
            nn.Linear(32,16),
            nn.LeakyReLU(0.01),
            nn.Linear(16,2))
        
    def forward(self,x):
        encoded = self.encoder(x)
        return encoded

class ConvClass4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global spatial summary
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 shape classes
        )
    
    def forward(self,x):
        decoded = self.encoder(x)
        return decoded
