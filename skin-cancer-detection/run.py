import os
import sys
import argparse
from app.inference import inference

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Inference')
    parser.add_argument('--image', type=str, default='data/images/example.jpg', 
                        help='Path to the input image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
        
    print(f"Running inference on {args.image}...")
    result = inference(args.image)
    print(f"Inference complete!")
    print(f"Classification: {result['classification']}")
    print(f"Output saved to: {result['output_path']}")

if __name__ == "__main__":
    main()