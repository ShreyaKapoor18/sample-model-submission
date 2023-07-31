from model_tools.check_submission import check_models
import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models
import torch

"""
Template module for a brain model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['resnet50']


def get_model(name):
    """
    This method fetches an instance of a brain model. The instance has to implement the BrainModel interface in the
    brain-score project(see imports). To get a detailed explanation of how the interface hast to be implemented,
    check out the brain-score project(https://github.com/brain-score/brain-score), examples section :param name: the
    name of the model to fetch
    :return: the model instance, which implements the BrainModel interface
    """
    assert name == 'resnet50'
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    wrapper = PytorchWrapper(identifier='resnet50', model=model, preprocessing=preprocessing)
    wrapper.image_size = 256

    return wrapper


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BrainModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_brain_models(__name__)
