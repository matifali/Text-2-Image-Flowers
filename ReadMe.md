EEE-543 Project

1) Set up a virtual environment
    python3 -m venv venv
1) Configure enevironment using the following command:
    pip install -r requirements.txt
3) Run the following command to train the model:
    python runtrain.py --cfg config.yml
4) Download the our pretaind model from the link:
https://drive.google.com/drive/folders/14iDDBKsjWUYArai755NiqGbRonAV5Hlp?usp=sharing
5) Place the model in directory './models/'
6) Opne config_test.yml and change the TEST key to appropriate value
   1 -- for test on the validation set
   2 -- for test on predifined cutsom captions
   3 -- for generating model summaary and architecture
4) Run the following command to test the model:
    python runtime.py --cfg config_test.yml
