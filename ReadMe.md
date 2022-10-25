# Text 2 Image Flowers

1) Set up a virtual environment
    python3 -m venv venv
1) Configure enevironment using the following command:
    pip install -r requirements.txt
3) Run the following command to train the model:
    python runtrain.py --cfg config.yml
4) Opne config_test.yml and change the TEST key to appropriate value
   1 -- for test on the validation set
   2 -- for test on predifined cutsom captions
   3 -- for generating model summaary and architecture
5) Run the following command to test the model:
    python runtime.py --cfg config_test.yml
    
Note: A checpoint is avaible at https://drive.google.com/file/d/1-b1MiUPppgY9zeNQCBqvhQIJRZV0BHBa/view?usp=sharing

Credits:
1) [Recurrent-Affine-Transformation-for-Text-to-image-Synthesis](https://arxiv.org/abs/2204.10482)
2) [RAT-GAN](https://github.com/senmaoy/Recurrent-Affine-Transformation-for-Text-to-image-Synthesis)

