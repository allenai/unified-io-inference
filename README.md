# UnifiedIO

This repo contains code to run models from our paper [Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks](https://arxiv.org/abs/2206.08916).

## Installation
Install [jax](https://github.com/google/jax#installation), note this might require manually installing
Cuda Toolkits and Cudnn toolkits if using GPUs.

Then install the supporting libraries with:

```
pip install -r requirements.txt
```

## Model weights
Model weights can be found on aws:
- XL: [https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/xl_1000k.bin](https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/xl_1000k.bin) (10.9gb)
- Large: [https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/large_1000k.bin](https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/large_1000k.bin) (3.2gb)
- Base: [https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/base_1000k.bin](https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/base_1000k.bin) (1.2gb)
- Small: [https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/small_1000k.bin](https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/small_1000k.bin) (0.6gb)

To download run:

```wget 
wget https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/small_1000k.bin -O small.bin
```

or download with aws-cli: 
```aws
aws s3 cp s3://ai2-prior-uio/public/model-weights-bin/small_1000k.bin small.bin 
```

## Usage
Download an image to test on:
```bash
wget https://farm2.staticflickr.com/1362/1261465554_95741e918b_z.jpg -O dbg_img.png
```

Then tasks can done using the `ModelRunner` class:

```python
from uio import runner
from PIL import Image

model = runner.ModelRunner("small", "small.bin")

with Image.open("dbg_img.png") as img:
  image = np.array(img.convert('RGB'))

# Answer a VQA question, note this might take over a minute the first time it is 
# called while the function is compiled by jax
output = model.vqa(image, "What color is the sofa?")
print(output["text"])  # Should print `green`
```

This example can be run end-to-end by `demo_script.py`. `ModelRunner` supports many more tasks, 
examples can be seen in the demo notebook.


`ModelRunner` also provides a lower-level API that can be called with arbitrary text/image output and 
can generate text/image outputs, as well supporting batch input

```python
out = model.run([image], ["What is the depth map of the image ?"], 
               output_text_len=1, generate_image=True, num_decodes=None)
depth_image = out["image"][0]
```

## Demo notebook
More tasks are shown in demo.ipynb, this requires additionally install jupyter and matplotlib:

```
pip install matplotlib notebook
```

Then it can be run with:

```python
jupyter notebook demo.ipynb
```


## Just-in-time compilation
By default `ModelRunner` compiles the underlying inference calls the first time they are used,
this results in faster performance at a one-time cost. This can be disabled by setting the
`compile` parameter to false. You can set the environment variable `JAX_LOG_COMPILES=1`
to see when a function is being compiled.

## Implementation Details
Running UnifiedIO on a task is a 4-step process:

1. Convert tasks inputs into (image_input, prompt) pairs, the image_input can be `None`.
This step is task-specific and involve things like selecting a prompt for the tasks 
or converting region locations into region location tokens that are then embedded in the prompt,
2. Preprocess these components, done by `utils.preprocess_image` and converting the input prompt into
tokens using a `T5Tokenizer`
3. Running the model on these pre-processed input, done in `model.py`. This produces text 
tokens and/or a 256x256 image as output. 
4. Post-process the results, this step is task-specific and can involve converting the output 
tokens into text or image locations and/or resizing/cropping the output image.

In `ModelRunner`, `run` does steps 2 and 3 and the task-specific methods do steps 1 and 4 
for various tasks.

The main neural network code itself can be found in `modules.Transformer` 

## Hardware requirements
We have run XL model on GPUs with 24GB of memory, lower memory GPUs should be able to run
the smaller models but might not be able to run the XL model. 

## Citation
If you use this codebase, please cite:

```
@article{lu2022unified,
  title={Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks},
  author={Lu, Jiasen and Clark, Christopher and Zellers, Rowan and Mottaghi, Roozbeh and Kembhavi, Aniruddha},
  journal={arXiv preprint arXiv:2206.08916},
  year={2022}
}
```
