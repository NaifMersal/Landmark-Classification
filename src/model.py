import torch
import torch.nn as nn




# define the CNN architecture
class DepthConv(nn.Module):
    def __init__(self, in_channels: int , out_channels:int, kernel_size:int, stride=1, padding=0,depth_count:int=2 ,growth_factor:int=1) -> None:
        # depth_count >2
        super().__init__()
        layers=[nn.Conv2d(in_channels=in_channels,out_channels=in_channels*growth_factor,kernel_size=kernel_size,stride=stride,padding=padding, groups=in_channels, bias=False)
                , nn.ReLU(inplace=True)]
        for _ in range(1,depth_count-1):
            layers.extend([nn.Conv2d(in_channels=in_channels*growth_factor,out_channels=in_channels*growth_factor,kernel_size=kernel_size,stride=stride,padding=padding, groups=in_channels*growth_factor, bias=False),
                           nn.BatchNorm2d(in_channels*growth_factor),
                           nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(in_channels=in_channels*growth_factor,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding, groups=out_channels, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)])   # this last one is not Depthwise  since groups=out_channels
        
        self.conv=nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
        

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.avgpoolP1=nn.AvgPool2d(2,2,1)
        self.avgpool=nn.AvgPool2d(2,2)
        self.Dconvs0=nn.Sequential(
           nn.Conv2d(3,64,7,stride=2,padding=3), # 112
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
       )       
        self.Dconvs1=nn.Sequential(
            DepthConv(64,64,3,padding=1,growth_factor=3 ,depth_count=2),
            nn.Dropout2d(dropout/4),
            nn.Conv2d(64, 32,1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout/4),
            nn.ReLU(inplace=True),
            )# then concate and Avgpool-> 56 
        
        self.Dconvs2=nn.Sequential(
            DepthConv(96,96,3,padding=1,growth_factor=3 ,depth_count=3),
            nn.Dropout2d(dropout/2),
            nn.Conv2d(96, 32,1),
            nn.Dropout2d(dropout/2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
         ) # then Avgpool -> 28  
        self.Dconvs3=nn.Sequential(
            DepthConv(128,128,3,padding=1,growth_factor=3 ,depth_count=3),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 32,1),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
         ) # then concate avgpool -> 14

        self.Dconvs4=nn.Sequential(
            DepthConv(160,160,3,padding=1,growth_factor=3 ,depth_count=3),
            nn.Dropout2d(dropout/2),
            nn.Conv2d(160, 32,1),
            nn.Dropout2d(dropout/2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
         )  # then concate Avgpool -> 7
        
        self.Dconvs5=nn.Sequential(
            nn.Conv2d(192,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 32,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
         )  # then concate avgpoolP1 -> 4
        
        self.Dconvs6=nn.Sequential(
            nn.Conv2d(224,224,1),
            nn.Dropout2d(dropout/4),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 32,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
         )  # then concate avgpool -> 2
        
        



    
        classifier_input=2*2*(224+32)
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input,num_classes),
        )

        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.Dconvs0(x)
        x=torch.cat((self.Dconvs1(x),x), dim=1)
        x=self.avgpool(x)
        x=torch.cat((self.Dconvs2(x),x), dim=1)
        x=self.avgpool(x)
        x=torch.cat((self.Dconvs3(x),x), dim=1)
        x=self.avgpool(x)
        x=torch.cat((self.Dconvs4(x),x), dim=1)
        x=self.avgpool(x)
        x=torch.cat((self.Dconvs5(x),x), dim=1)
        x=self.avgpoolP1(x)
        x=torch.cat((self.Dconvs6(x),x), dim=1)
        x=self.avgpool(x)




        x=self.fc(x)
        return x



from torchvision.models import densenet121
class Model(nn.Module):
    def __init__(self, num_classes=50, trained=False, dropout: float = 0.4) -> None:

        super().__init__()

        
        classifier_input=1000
        self.model=nn.Sequential(
            densenet121(trained),
            nn.Linear(classifier_input,int(classifier_input/4)),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout), 
            nn.Linear(int(classifier_input/4), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
