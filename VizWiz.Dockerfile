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

RUN apt-get -y install openjdk-8-jdk
RUN java -version
RUN apt-get -y install git tree nano

WORKDIR /root/vizwiz/
RUN git clone https://github.com/Yinan-Zhao/vizwiz-caption.git
WORKDIR /root/vizwiz/vizwiz-caption
RUN apt-get -y install unzip 
RUN bash ./get_stanford_models.sh
#RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip

#RUN export danny=$(find /root -name 'stanford-corenlp-3.6.0.jar')
#RUN export danny=$(find /root/vizwiz -name '*.jar')
#RUN echo $danny
#RUN fjdsfdsfdsf

COPY eval-vizwiz.py .
WORKDIR /root

COPY vizwiz.yml .
RUN bash -c "conda env create -f vizwiz.yml"
RUN bash -c ". activate vizwiz && pip install --upgrade pip \
 && python3 -m pip install matplotlib notebook numpy==1.23.5"

WORKDIR /root/vizwiz/vizwiz-caption/annotations
# TODO create small sample annotations file for testing
COPY val.json val.json

#ENV JAVA_HOME=
#EXISTS: /root/vizwiz/vizwiz-caption/vizwiz_eval_cap/tokenizer/stanford-corenlp-3.4.1.jar
ENV CLASSPATH=.;/root/vizwiz/vizwiz-caption;/root/vizwiz/vizwiz-caption/vizwiz_eval_cap/tokenizer
WORKDIR /root/vizwiz/vizwiz-caption

RUN bash -c ". activate vizwiz && python3 eval-vizwiz.py"
WORKDIR /root

COPY . .
RUN bash -c ". activate uioi && export PYTHONPATH=/root:/root/uio && python ./uio/test/check.py"
ENV OUTPUT_FILE=/output/output.txt
ENV IMAGE_DIR=/images
ENV SAMPLE_COUNT=1
ENTRYPOINT bash -c ". activate uioi && python ./caption-vizwiz.py xl xl.bin $VIZWIZ_FILE $IMAGE_DIR $OUTPUT_FILE $SAMPLE_COUNT"
