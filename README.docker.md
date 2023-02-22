
## Docker
To build a docker image:
```bash
docker build -t unified-io-inference .
```

To run captioning of the CC12M dataset using Unified-IO:
```bash
docker run -it --gpus=1 -e WEBDATASET_FILE=/input/00000.tar -v /nas/gaia02/data/paper2023/cc12m/images:/input -v /nas/gaia02/users/napiersk/github/feb-14/unified-io-inference/output:/output -e SAMPLE_COUNT=500 unified-io-inference
```
