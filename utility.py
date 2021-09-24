# python 3.8.5

# Imports here
import torch                                         # 1.7.0
from torchvision import datasets, transforms
from PIL import Image                                # 7.2.0
import numpy as np                                   # 1.19.1

def load_data(data_dir):
    ''' Loads image data for model training and validation
    '''

    print(f'Loading data from {data_dir}...\n')

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(90),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)

    # Use the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    class_to_idx = train_dataset.class_to_idx

    print(f'Data loaded from {train_dir} and {valid_dir}.\n')

    return train_dataloader, valid_dataloader, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    print(f'Processing image at {image}...\n')

    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)

    # Resize to 256 pixels on shortest dimension, even number on largest dimension
    if image.width >= image.height:
        dim = (2 * int(128 * image.width / image.height), 256)
    else:
        dim = (256, 2 * int(128 * image.height / image.width))
        
    image = image.resize(dim)

    # Crop centre 224x224 pixels
    h_crop = (image.width - 224) / 2
    v_crop = (image.height - 224) / 2

    image = image.crop((h_crop, v_crop, image.width-h_crop, image.height-v_crop))

    # Convert and normalise colour channel values
    np_image = ((np.array(image) / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # Transpose dimensions
    np_image = np_image.transpose((2, 0, 1))

    tensor_image = torch.from_numpy(np_image).float()

    print('Image processed.\n')

    return tensor_image