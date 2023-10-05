import pytest
import torch
import torch.optim as optim
from Models import Optimizers 

@pytest.fixture
def optimizers():
    # Create a list of optimizers for testing
    num_optimizers = 3
    list_of_optimizers = [optim.SGD([torch.nn.Parameter(torch.rand(10))], lr=0.1) for _ in range(num_optimizers)]
    return Optimizers(list_of_optimizers)

def test_zero_grad(optimizers):
    # Not implemented yet
    pass
            
def test_step(optimizers):
    # Perform step on the Optimizers
    optimizers.step()

    # Check if step is called on each optimizer
    for opt in optimizers.optimizers:
        # check if requreis grad is true
        assert opt.param_groups[0]['params'][0].requires_grad == True

