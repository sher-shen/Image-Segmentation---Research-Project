# Image-Segmentation---Research-Project

## Project Structure

This project comprises several key components distributed in different folders for a clear organization and easy replication of experiments.

### Tried Architectures (`essaie_architecture`)

We have tested various neural network architectures for our image segmentation task. To run a specific architecture, simply rename the desired folder to `architecture`.

The available architectures include:
- ResNet18
- ResNet101
- DenseNet121
- FCN (Fully Convolutional Network)
- EfficientNet (versions b0 and b1 with PPM module)

### Loss Function Trials (`essaieloss`)
Several loss functions were experimented with to refine our model. To test them, rename the chosen loss function folder to `loss`.

The loss functions explored are:

- Binary Cross-Entropy (BCE)
- Dice Loss
- Focal Loss
- Boundary Loss
- Generalised Dice Loss
- Combination of BCE and Dice Loss

### Baseline Model (`best_model`)

The `best_model` folder contains the model that achieved the best results during our tests. You can use it as a benchmark for performance.

### Training Loss Visualization (`plot_loss_exemple`)

The `plot_loss_exemple` folder contains scripts for visualizing training loss curves. This can be useful for analyzing model behavior during training.

## Usage

TTo run the segmentation model, use the following command:

```bash
python -m src.main --train
```

Before running the segmentation script, ensure that you have renamed the required architecture and loss function files. Simply name the architecture file you want to use as `architecture.py`, and the loss function file as `loss.py`. The currently employed `architecture.py` and `loss.py` represent the optimal combination, which is EfficientNet b1 + PPM, with the loss function being a combination of BCE + Dice. There's no need to change any other parts; just execute `python -m src.main --train`, and you will find the results in the `submission` folder within the `baseline` directory, which should contain many `.npy` files.

Additionally, if you wish to visualize the training loss, you can uncomment the last few lines in the final section of the script in the `instance segmentation` . This will generate and save an image of the loss curve.

If you wish to compare the predicted data with the actual data, you can use the command:

```bash
python -m src.main --evaluate
```

This will return an mIoU score, which is a standard metric for evaluating the quality of image segmentation models.
## Acknowledgements
This project is built upon the models and frameworks provided at [EY](https://gitlab.com/ey_datakalab/ihm_instance_segmentation). 
