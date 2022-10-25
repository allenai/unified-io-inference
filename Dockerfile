FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
LABEL name="unified-io-inference"

WORKDIR /root/.conda
WORKDIR /root
RUN apt-get update && apt-get -y install wget nano
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH=/opt/conda/bin:${PATH}
RUN bash -c "conda update -n base -c defaults conda"

RUN wget -nv https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/xl_1000k.bin \
 -O xl.bin
RUN wget -nv https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/large_1000k.bin \
 -O large.bin
RUN wget -nv https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/base_1000k.bin \
 -O base.bin
RUN wget -nv https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/small_1000k.bin \
 -O small.bin
RUN wget -nv https://farm2.staticflickr.com/1362/1261465554_95741e918b_z.jpg -O dbg_img.png

COPY uioi.yml .
RUN bash -c "conda env create -f uioi.yml"
COPY requirements.txt .
RUN bash -c ". activate uioi && pip install --upgrade pip \
 && pip install --upgrade "jax[cuda]" \
 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
 && python3 -m pip install -r requirements.txt"

COPY . .
RUN bash -c ". activate uioi && python ./uio/test/check.py"
ENV INPUT_FILE=demo.list
ENTRYPOINT bash -c ". activate uioi && python ./run.py xl xl.bin $INPUT_FILE"
