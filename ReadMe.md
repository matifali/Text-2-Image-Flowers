# Text 2 Image Flowers

1) Set up a virtual environment
```console
python3 -m venv venv
```

2) Configure enevironment using the following command:
```console
pip install -r requirements.txt
```

3) Run the following command to train the model:
```console
python runtrain.py --cfg config.yml
```

4) Open [`config_test.yml`](https://github.com/matifali/Text-2-Image-Flowers/blob/88adb45342dd90010d93f18565287c695b9747bd/config_test.yml#L8) and change the `TEST` key to appropriate value:
   - `1` for test on the validation set
   - `2` for test on predifined cutsom captions
   - `3` for generating model summary and architecture

5) Run the following command to test the model:
```console
python runtime.py --cfg config_test.yml
```

Note: A checpoint is avaible at https://drive.google.com/file/d/1-b1MiUPppgY9zeNQCBqvhQIJRZV0BHBa/view?usp=sharing

Credits:
1) [Recurrent-Affine-Transformation-for-Text-to-image-Synthesis](https://arxiv.org/abs/2204.10482)
2) [RAT-GAN](https://github.com/senmaoy/Recurrent-Affine-Transformation-for-Text-to-image-Synthesis)

