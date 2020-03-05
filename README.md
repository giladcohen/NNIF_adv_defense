# NNIF_adv_defense
Detection of adversarial examples using influence functions and nearest neighbors

### Installation
This repo supports python 2.7

Install the dependencies:

```sh
$ cd NNIF_adv_defense
$ pip install -r requirements.txt
```

### Training

To train a model on a dataset, run:
```sh
$ python NNIF_adv_defense/train.py --dataset cifar10
```
You can choose between CIFAR-10/100 or SVHN. Also, you can also contol the number of epoches (--nb_epochs) and the batch size (--batch_size).
This script also seperates the dataset into train/val/test of 49k/1k/10k. For SVHN we randomely select 50k samples from its training set and 10k samples from its test set.

### Evaluating and Attacking
The script to evaluate and attack your pretrained network is attack.py. Run with:
```sh
$ python NNIF_adv_defense/attack.py --dataset cifar10 --set val --attack cw
```
This will attack your network using the Carlini-Wagner attack, on a validation set (1k samples out of the 50k). After the attack finishes, you need to repeat it also with the test set (--set test).
The available attacks are: fgsm, jsma, deepfool, and cw

### Evaluating and Attacking - faster method
-----STAGE A-----

The code above runs very slowly and takes days to run for a single dataset. To make the code run in a realistic time, run:
```sh
$ python NNIF_adv_defense/calc_hvp.py --dataset cifar10 --set val --attack cw
```
This will only calculate the Hessian inverse approximation (see https://arxiv.org/abs/1703.04730) and not the entire I_up_loss. It is highly recommended to use GPUs for this run.

-----STAGE B-----

After STAGE A completes, run this script:
```sh
$ CUDA_VISIBLE_DEVICES='' python NNIF_adv_defense/calc_scores.py --dataset cifar10 --set val --attack cw --num_threads 4
```
This scripts assumes you have at least 4 CPU cores on your machine. set --num_threads to the actual number of your CPU cores. You can check this via running
```sh
$ lscpu | egrep "CPU\(s\)" -m1
```
Again, run this code for both val and test datasets

### Detection Adversarial Examples
First, collect the features:
```sh
$ python NNIF_adv_defense/extract_characteristics.py --dataset cifar10 --attack cw --characteristics nnif --max_indices 50
```

Next, train and evaluate the Logistic Regression detector using the val/test features:
```sh
$ python NNIF_adv_defense/detect_adv_examples.py --dataset cifar10 --attack cw --characteristics nnif --max_indices 50
```

### White-Box attack
First you must create scores (using the attack.py script) for any attack (to generate the .../pred/score.npy for the natural image). Next, run:
```sh
$ python NNIF_adv_defense/white_box_attack.py --dataset cifar10 --set val
```
Once again, before running the detection scripts you must have adversarial images for both the val set and test set (so run also with --set test)
