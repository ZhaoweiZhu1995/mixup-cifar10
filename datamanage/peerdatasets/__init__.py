from .MNIST import MNIST, FashionMNIST
from .CIFAR import CIFAR10, CIFAR100
from .CIFAR10H_5K import CIFAR10H_5K
from .CIFAR10H import CIFAR10H
from .ManfredDemo import ManfredDemo
from .YangDemo import YangDemo
from .View import View
from .SVHN import SVHN


__all__ = [
    "MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "ManfredDemo", "YangDemo", "View", "SVHN",
    "CIFAR10H_5K", "CIFAR10H"
]