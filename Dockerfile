FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
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

RUN bash -c ". activate uioi && pip install matplotlib notebook numpy==1.23.5 nvidia-cudnn-cu11==8.6.0.163 setuptools wheel spacy webdataset && python3 -m spacy download en_core_web_sm"

ENV PYTHONPATH=/root/uio

COPY . .
RUN bash -c ". activate uioi && export PYTHONPATH=/root:/root/uio && python ./uio/test/check.py"
ENV OUTPUT_FILE=/output/output.txt
ENV SAMPLE_COUNT=1
ENTRYPOINT bash -c ". activate uioi && python ./caption.py xl xl.bin $WEBDATASET_FILE $OUTPUT_FILE $SAMPLE_COUNT"
