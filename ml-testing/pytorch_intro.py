import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Display (a single) image from a (H, W, 3) torch.Tensor img.
def imshow(img, show=True):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='bicubic')
    if show:
        plt.show()


# Display two torch.Tensors img and output as images, side by side.
def compare_img(img, output):
    p = torch.as_tensor([img[0].detach().numpy(), output[0].detach().numpy()])
    imshow(torchvision.utils.make_grid(p).detach(), show=False)
    plt.show()


# A first simple neural network class. Subclasses nn.Module.
class simple_linear_net(nn.Module):

    # Define the constituents of the net: the layers of the nn, weighs and biases.
    # Special case of batch_size = 1: batch dimension non-existent.
    def __init__(self):
        super(simple_linear_net,
              self).__init__()  # Usage of nn.Parameter requires initialization of superclass nn.Module

        # Since input image is flattened (length 32*32), and the net only consist of one layer, we want the
        # second dimension of weighs matrix to be equal to the number of classes, thus weighs is 1024*10 as
        # weighs @ x = [1024 x (1024 x 10)] = [10]
        self.weighs = nn.Parameter(torch.randn(32 * 32, 10)) / np.sqrt(32)  # Xavier
        self.bias = nn.Parameter(torch.zeros(32))

    # One iteration of the data flowing through the network: maps input (flattened) image tensor x to output tensor of the nn.
    # This case, perform a linear transformation of x. '@' denotes the dot product in R^n, where n = dim x.
    def forward(self, x):
        # operations on x
        return x @ self.weighs + self.bias


def run_pipeline(batch_size, shuffle):
    # TODO: remove when performing actual testing
    torch.manual_seed(1)

    # Choice of transformations of data, .Compose() creates a composition of transformations:
    # .toTensor() maps a PIL image to a triplet (channels, height, width).
    # NOTE: one can use nn.Sequential to make a 'scriptable transform' i.e. (?) a 'transformation layer' in the nn.
    transform = transforms.Compose([transforms.ToTensor()])

    # Download train data, as an 'abstract' class datasets.CIFAR10 (i.e. contains more information than pure numberd
    # Put the data in directory './data' (here + /data).
    train_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Create (iterable) DataLoader object which can fetch the actual data from the abstract .dataset class. Includes
    # various settings for batches, shuffling etc.
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=shuffle)

    # Actual data is fetched using the iterator method __next__. Since DataLoader
    # is iterable, we can create an Iterator iter(DataLoader) from it.
    train_data_iter = iter(train_loader)

    # From the iterator, we can iterate through the images and labels
    images, labels = train_data_iter.next()
    images_flat = images.view(3, 32 * 32)

    # Make input track its gradients, i.e. enable automatic backpropagation
    images.requires_grad_()

    # Construct net and let image go through it once
    net = simple_linear_net()
    # output = net.forward(images)

    print(images_flat.shape)
    print((net.weighs).shape)
    print((torch.matmul(images_flat, net.weighs)))
    # Visually compare output image with original image
    # compare_img(images, output)

    return


if __name__ == '__main__':
    # Initialize constants
    BATCH_SIZE = 1
    SHUFFLE = False

    run_pipeline(BATCH_SIZE, SHUFFLE)

"""
# Creates input (leaf) tensor x of shape NxN on which operations are tracked
x = torch.ones(N, N, requires_grad=True)

# Do operations on leaf tensor: i.e. create non-leaf tensors y, out.
y = x + 2
loss = (y*y).mean()

# N**2 number of matrix elements. loss == o
# o_i = 1/N**2 sum_i^N y_i**2 = 1/N**2 sum_i^N (x_i + 2)**2
#
# do/dx_i = 2/N**2 (x_i + 2), do/dx_i|(x_i=1) = 2 / N**2 * 3 = 3/2 if N=2

# Backwards differentiate (backprop) from out->x, i.e. calculate the gradients
# in the graph and save in the .grad attribute of leaf tensors (else .retain_grad())
print(loss.backward())

# Display 'forward' gradient d(loss)/dx (stored as attribute of leaf node x),
# calculated after backpropagation is conducted, through the invocation of loss.backward().
print(x.grad)
"""
