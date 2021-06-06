# CIFAR-10

#### &nbsp; CIFAR-10 classifier with PyTorch

&nbsp; I implemented a custom model to classify CIFAR-10 with PyTorch. <br/>
&nbsp; This model achieve 91.34% test accuracy. <br/><br/>

## Requirements

&nbsp; Python 3.8 or later with all [requirements.txt](https://github.com/Igoc/CIFAR-10/blob/master/requirements.txt), including `pytorch >= 1.8`.

```
pip install -r requirements.txt
```

<br/>

## Usage

```
Usage: python "Training.py" [-h] [-v] [--checkpoint] [--cpu] [--seed S]
                            [--batchsize N] [--epochs N] [--lr LR]
```

```
Optional Arguments:
  -h, --help     Show help message and exit
  -v, --verbose  Enables verbosity
  --checkpoint   Enables checkpoint
  --cpu          Disables CUDA training
  --seed S       Random seed (default: 0)
  --batchsize N  Input batch size (default: 64)
  --epochs N     Number of epochs to train (default: 150)
  --lr LR        Learning rate (default: 0.001)
```