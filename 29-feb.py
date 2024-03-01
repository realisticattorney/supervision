import argparse
import json
import os
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input file and model.")
    parser.add_argument("input_file", help="Input video file name")
    parser.add_argument("--model", help="Model to use for processing", required=True)
    return parser.parse_args()

def load_configuration(input_file):
    with open('config.json', 'r') as file:
        configs = json.load(file)
    return configs.get(input_file, None)

def generate_output_filename(input_file, model):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return f"./outputvids/{base_name}_{model}_output.mp4"

def get_model_names():
    return [os.path.splitext(os.path.basename(pt))[0] for pt in glob.glob("*.pt")]

def main():
    args = parse_arguments()
    if args.model not in get_model_names():
        print(f"Model {args.model} not found. Available models: {', '.join(get_model_names())}")
        return

    config = load_configuration(args.input_file)
    if not config:
        print(f"No configuration found for {args.input_file}")
        return

    output_filename = generate_output_filename(args.input_file, args.model)
    
    # Now, use `config` for processing with the chosen model and save output to `output_filename`
    print(f"Processing {args.input_file} with model {args.model}...")
    print(f"Output will be saved to {output_filename}")

if __name__ == "__main__":
    main()
