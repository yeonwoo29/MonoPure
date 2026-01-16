
# MonoPure
    
1. Install pytorch and torchvision matching your CUDA version:
    ```bash
    # For example, We adopt torch 1.9.0+cu111
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    
2. Install requirements and compile the deformable attention:
    ```bash
    pip install -r requirements.txt

    cd lib/models/monopure/ops/
    bash make.sh
    
    cd ../../../..
    ```
 
4. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```bash
    │Monopure/
    ├──...
    │data/kitti/
    ├──ImageSets/
    ├──training/
    │   ├──image_2
    │   ├──label_2
    │   ├──calib
    ├──testing/
    │   ├──image_2
    │   ├──calib
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monopure.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monopure.yaml` and indicate the GPU in `train.sh`:
  ```bash
  bash train.sh configs/monopure.yaml > logs/monopure.log
  ```
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monopure.yaml`:
  ```bash
  bash test.sh configs/monopure.yaml
  ```

## Acknowlegment
This repo benefits from the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP).
