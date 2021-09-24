# python 3.8.5

# Imports here
import argparse
import json
from model_helper import *
from utility import *

def main():
    parser = argparse.ArgumentParser(
        description='Predictor for flower identification neural network',)
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', type=int, default=1,
        help='Select number of most likely classes to display (default: %(default)s)')
    parser.add_argument('--category_names', default='cat_to_name.json',
        help='Select category to names mapping file (default: %(default)s)')
    parser.add_argument('--gpu', action='store_true',
        help='Flag to use GPU if available (default: %(default)s)')
    args = parser.parse_args()

    # Restore model
    model, device = load_checkpoint(args.checkpoint, args.gpu)

    # Process image
    image = process_image(args.image_path)

    # Predict image class
    probs, classes = predict(image, model, device, args.top_k)

    # Convert classes to class names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[class_num] for class_num in classes ]

    for i in range(0, len(probs)):
        print(f'{class_names[i]}: {probs[i]*100}%')

if __name__ == "__main__":
    main()