import torchvision.transforms as transforms


from .random_crop import RandomCrop


def create(random_crop=None, horizontal_flip=False):
    """
    Create a pytorch transformation for image augmentations.
    All-in-one package.
    """
    transformations = []

    if random_crop:
        transformations.append(RandomCrop(
            size=(random_crop['width'], random_crop['height']),
            padding=random_crop.get('padding', 0),
            padding_mode=random_crop.get('padding_mode', 'constant')
        ))

    if horizontal_flip:
        transformations.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transformations)

