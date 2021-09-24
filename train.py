# python 3.8.5

# Imports here
import argparse
from model_helper import *
from utility import *

def main():
    parser = argparse.ArgumentParser(
        description='Trainer for flower identification neural network',)
    parser.add_argument('data_dir',
        help='Data directory must contain "train" and "valid" subdirectories')
    parser.add_argument('--save_dir', default='',
        help='Select subdirectory for saved checkpoint (default: same as --arch)')
    parser.add_argument('--arch', default='squeezenet', choices=['squeezenet', 'vgg13'],
        help='Select squeezenet or vgg13 architecture (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Select learning rate (default: %(default)s)')
    parser.add_argument('--hidden_units', type=int, default=4096,
        help='Select number of hidden units - supported for vgg13 only (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=4,
        help='Select number of epochs to train for (default: %(default)s)')
    parser.add_argument('--gpu', action='store_true',
        help='Flag to use GPU if available (default: %(default)s)')
    args = parser.parse_args()

    # Load data
    train_dataloader, valid_dataloader, class_to_idx = load_data(args.data_dir)
    
    # Build model
    model = build_model(args.arch, args.hidden_units)

    # Train model
    model = train_model(model,
                        args.learning_rate,
                        args.epochs,
                        args.gpu,
                        train_dataloader,
                        valid_dataloader)

    # Save checkpoint
    save_model(model, args.arch, args.hidden_units, args.save_dir, class_to_idx)

if __name__ == "__main__":
    main()