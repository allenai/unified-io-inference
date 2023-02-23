from argparse import ArgumentParser
import json
from pathlib import Path
from uio.runner import IMAGE_TAGGING


def process_imagenet_classes(classes_file: Path):
	classes_list = []
	with classes_file.open('r') as fp:
		for line in fp.readlines():
			classes_str = line[line.index(' '):].strip()
			classes = [c.strip() for c in classes_str.split(',')]
			classes_list.extend(classes)

	return {'classes': classes_list}

PREPROCESS_FUNCTIONS = {'imagenet': process_imagenet_classes}

def main():
	parser = ArgumentParser(__doc__)
	parser.add_argument(
			"--input-file", required=True, type=Path, help="Input file containing class names."
	)
	parser.add_argument(
			"--input-format", required=True, type=str,choices=PREPROCESS_FUNCTIONS.keys(), help="Input file format."
	)

	parser.add_argument(
			"--output-file",
			required=True,
			type=Path,
			help="Output file to write the classes dict.",
	)

	args = parser.parse_args()
	input_file = args.input_file
	input_format = args.input_format
	output_file = args.output_file
	
	prep_function = PREPROCESS_FUNCTIONS[input_format]
	classes_dict = prep_function(input_file)
	with output_file.open('w') as fp:
		json.dump(classes_dict,fp)

if __name__ == "__main__":
	main()
