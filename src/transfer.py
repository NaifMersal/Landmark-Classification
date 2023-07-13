import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")
    
    # detrmine the last layer
    classifier='fc' if hasattr(model_transfer,'fc') else 'classifier'  
    
    # detrmine num_features( this not a good way but it works)
    try:
        setattr(model_transfer,classifier, nn.Linear(2,1))
        x=torch.zeros(1,3,224,224)
        model_transfer(x)
    except Exception as e:
        error=str(e)
        start_index=error.index('(')
        end_index=error[start_index:].index('a')
        in_features=int(error[start_index:start_index+end_index].split('x')[1])



    # Freeze all parameters in the model
    # HINT: loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    for param in model_transfer.parameters():
        param.requires_grad = False
    



    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs  = in_features

    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs
    fc=nn.Sequential(
        nn.Dropout1d(0.2),
        nn.Linear(num_ftrs,int(num_ftrs/4)),
        nn.Dropout(0.1),
        nn.ReLU(inplace=True),
        nn.Linear(int(num_ftrs/4), n_classes),
        ) 
    setattr(model_transfer,classifier, fc)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
