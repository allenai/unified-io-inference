
## Docker
To build a docker image:
```bash
docker build -t unified-io-inference .
```
To run the docker demo:
```
docker run -it --gpus=1 unified-io-inference
INFO:absl:Setting up model...
...
INFO:absl:Model is ready
INFO:absl:Running model text_inputs=['what color is the sofa?']
green
```

To run a list of queries construct an input file where each line is a file path
and a text input, separated by `:`.  See example: [demo.list](https://github.com/isi-vista/unified-io-inference/blob/docker-build/demo.list)

Prepare a directory containing image files.  `cd` to that directory.
The steps below will write example input files and docker execution with the 
host images mounted to the `/image-data` directory.

```
ls -1 | grep -E 'jpg|png' > files.txt
awk '{print "/image-data/" $0 ":What-does-the-image-describe?"}' ./files.txt > caption.txt
awk '{print "/image-data/" $0 ":Locate all objects in the image."}' ./files.txt > locate.txt

#Choose an input file to process:
export INPUT_FILE=[caption.txt or locate.txt or other]
export HOSTPATH=$(pwd)

docker run -it --gpus=1 -e INPUT_FILE=/image-data/${INPUT_FILE} \ 
  -v /${HOSTPATH}:/image-data unified-io-inference
```
