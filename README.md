# CIPLAB Deepfake Detection Project

## Setup
1. Clone the repository:

    ```bash
    git clone https://github.com/cygbbhx/deepfake/
    cd deepfake
    ```

2. Setting up docker:

    ```bash
    docker pull ciplab/sohyun_deepfake:latest
    ```
    - or use `DockerFile` in the repo

3. Configuring datapaths
- Modify paths inside `config/data_paths.json` according to your settings.

## Training

```bash
python train.py path_to_your_config.yaml
```

### Testing

```bash
python test.py path_to_your_checkpoint_experiment_directory
```
- The test code automatically finds for `best.pt` inside the checkpoint directory. If you add additional argument of the path to weights, it will use the weight for testing.
- The test results (logs) will be saved inside the checkpoint directory you have set.

### Extracting Features

```bash
python feature_extraction.py -w path_to_model_weights
```

### Visualizing Features

```bash
python visualize_all.py path_to_feature_directory
```
