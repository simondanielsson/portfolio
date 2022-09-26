from torchvision.models import resnet18
from torch import nn

class TripletNet(nn.Module):
  """Composite network: MLP attached to ResNet"""

  def __init__(self, out_dim):
    super(TripletNet, self).__init__()
    self.singlet_net = SingletNet(out_dim)

  def forward(self, a, p, n):
    a, p, n = self.singlet_net(a), self.singlet_net(p), self.singlet_net(n)

    return a, p, n


class SingletNet(nn.Module):
  """Maps input image through ResNet18 and MLP (2 layers) into representation space"""

  def __init__(self, out_dim):
    super(SingletNet, self).__init__()

    self.resnet = ResNetModified() 
    self.fc1 = nn.Linear(self.resnet.out_dim, 784)
    self.activation_fn = nn.ReLU()
    self.fc2 = nn.Linear(784, out_dim)

  def forward(self, x):
    x = self.resnet(x)
    x = self.activation_fn(self.fc1(x))
    x = self.fc2(x)

    return x


class ResNetModified(nn.Module):
  """ResNet without the trailing fully connected layer"""

  def __init__(self):
    super(ResNetModified, self).__init__()
    resnet = resnet18(pretrained=True)

    # Remove last fc layer of original ResNet
    self.layers = nn.Sequential(
        *list(resnet.children())[:-1]
    )

    self.out_dim = resnet.fc.in_features # 512

  def __forward__(self, x):
    x = self.layers(x)

    return x
