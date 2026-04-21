# data/transforms.py

from torchvision import transforms


def get_basic_image_transform(image_size=224):
    """
    Basic transform for inspecting CelebA images or using images
    outside of CLIP preprocessing.
    Output: tensor in [0, 1]
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_reconstruction_transform(image_size=224):
    """
    Transform for reconstruction targets.
    This is what your decoder will try to reproduce.
    Output: tensor in [0, 1]
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


"""
A note here: I am not putting CLIP preprocess directly in this file yet, because that depends on the model you load later.
Once we build the CLIP wrapper, we can pass CLIP’s own preprocess into the dataset.
"""