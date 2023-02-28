
## Docker
To build a docker image:
```bash
docker build -t unified-io-inference .
```

To build using named Dockerfile:
```bash
docker build -t unified-io-inference:vizwiz -f VizWiz.Dockerfile .
```

To run captioning of the CC12M dataset using Unified-IO:
```bash
docker run -it --gpus=1 -e WEBDATASET_FILE=/input/00000.tar -v /nas/gaia02/data/paper2023/cc12m/images:/input -v /nas/gaia02/users/napiersk/github/feb-14/unified-io-inference/output:/output -e SAMPLE_COUNT=500 unified-io-inference
```

To run vizwiz:
```bash
docker run -it --gpus=1 -v /nas/gaia02/data/paper2023/vizwiz/data/images/test:/images \
-v /nas/gaia02/data/paper2023/vizwiz/data/images/test:/root/vizwiz/vizwiz-caption/images \
-v /nas/gaia02/data/paper2023/vizwiz/data/annotations:/root/vizwiz/vizwiz-caption/annotations \
-v /nas/gaia02/data/paper2023/vizwiz/data/annotations:/input \
-e VIZWIZ_FILE=/input/train.json \
--entrypoint /bin/bash unified-io-inference:vizwiz
```
