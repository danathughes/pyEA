## cnn_genes
##
## Possible CNN layers encoded as individual genes
##

INPUT = "INPUT"
CONV1D = "CONV1D"
CONV2D = "CONV2D"
POOL1D = "POOL1D"
POOL2D = "POOL2D"
FULLY_CONNECTED = "FULLYCONNECTED"
OUTPUT = "OUTPUT"

from .InputGene import InputGene
from .OutputGene import OutputGene
from .FullyConnectedGene import FullyConnectedGene
from .Conv1DGene import Conv1DGene
from .Conv2DGene import Conv2DGene
from .Pool1DGene import Pool1DGene
from .Pool2DGene import Pool2DGene
from .AbstractGene import AbstractGene
from .DummyGene import DummyGene