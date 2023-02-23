from argparse import ArgumentParser
from pathlib import Path
from uio.runner import IMAGE_TAGGING

DEFAULT_QUESTION = IMAGE_TAGGING

def preprocess_images(images_dir: Path):
	uio_file_contents = []
	img_filetypes = ['*.jpg','*.png']
	for filetype in img_filetypes:
			for input_file in images_dir.rglob(filetype):
				uio_line = f"{input_file.absolute()}:{DEFAULT_QUESTION}"
				uio_file_contents.append(uio_line)
	return uio_file_contents 

def main():
	parser = ArgumentParser(__doc__)
	parser.add_argument(
			"--input-dir", required=True, type=Path, help="Input directory containing images."
	)

	parser.add_argument(
			"--output-file",
			required=True,
			type=Path,
			help="Output file to write the unified inputs.",
	)

	args = parser.parse_args()
	
	uio_file_contents = preprocess_images(args.input_dir)
	with args.output_file.open('w') as fp:
		fp.write('\n'.join(uio_file_contents))

if __name__ == "__main__":
	main()
