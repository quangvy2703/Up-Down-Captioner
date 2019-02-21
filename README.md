# Extracting image features with Faster-RCNN

### Requirements: software

0. **`Important`** Please use the version of caffe provided as a submodule within this repository. It contains additional layers and features required for captioning.

1.  Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

    **Note:** Caffe *must* be built with support for Python layers and NCCL!

    ```make
    # In your Makefile.config, make sure to have these lines uncommented
    WITH_PYTHON_LAYER := 1
    USE_NCCL := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    ```
3.  Nvidia's NCCL library which is used for multi-GPU training https://github.com/NVIDIA/nccl

### Requirements: hardware

By default, the provided training scripts assume that two gpus are available, with indices 0,1. Training on two gpus takes around 9 hours. Any NVIDIA GPU with 8GB or larger memory should be OK. Training scripts and prototxt files will require minor modifications to train on a single gpu (e.g. set `iter_size` to 2).

### Installation
You can follow the installing steps base on https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/.
But remember using caffe version on this repo.
All instructions are from the top level directory. To run the demo, should be only steps 1-4 required (remaining steps are for training a model).

1.  Clone the Up-Down-Captioner repository:
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/quangvy2703/Up-Down-Captioner.git
    ```

    If you forget to clone with the `--recursive` flag, then you'll need to manually clone the submodules:
    ```Shell
    git submodule update --init --recursive
    ```

2.  Build Caffe and pycaffe:
    ```Shell
    cd ./external/caffe

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

3.  Add python layers and caffe build to PYTHONPATH:
    ```Shell
    cd $REPO_ROOT
    export PYTHONPATH=${PYTHONPATH}:$(pwd)/layers:$(pwd)/lib:$(pwd)/external/caffe/python
    ```
    
4.  Build Ross Girshick's Cython modules (to run the demo on new images)
    ```Shell
    cd $REPO_ROOT/lib
    make
    ```
    
5.  Download Stanford CoreNLP (required by the evaluation code):
    ```Shell
    cd ./external/coco-caption
    ./get_stanford_models.sh
    ```

6.  Download the MS COCO train/val image caption annotations. Extract all the json files into one folder `$COCOdata`, then create a symlink to this location:
    ```Shell
    cd $REPO_ROOT/data
    ln -s $COCOdata coco
    ``` 

7.  Pre-process the caption annotations for training (building vocabs etc).
    ```Shell
    cd $REPO_ROOT
    python scripts/preprocess_coco.py
    ``` 


### Usage

1. `scripts/generate_baseline_mine.py`
    To extract image features, run above python script with following arguments:
      --image_folder, path to the folder contain image 
      --output_file, path to the .tsv file you want to save image features.
    Example
    ```
    python scripts/generate_baseline_mine.py --image_folder data/images/coco_train/ --output_file data/features/train.tsv
    ```

2. In the case GPU out of memory, change the following code line:
   Reduce max number boxes proposal
   ```
   fg['TEST']['RPN_POST_NMS_TOP_N'] = 150  # Previously 300 for evaluations reported in the paper
   ```
   Decrease max boxes
   ```
   MIN_BOXES = 10
   MAX_BOXES = 100
   ```
3. Enjoy your awnsome features.
