# Learnable Extended Activation Function (LEAF) for Deep Neural Networks

Implementation of the experiment as published in the paper "Learnable Extended
Activation Function for Deep Neural Networks" by
Yevgeniy Bodyanskiy and Serhii Kostiuk.

## Running experiments

1. NVIDIA GPU recommended with at least 2 GiB of VRAM.
2. Install the requirements from `requirements.txt`.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment variables.
4. Use the root of this repository as the current directory.
5. Add the current directory to `PYTHONPATH` so it can find the modules

This repository contains a wrapper script that sets all the required
environment variables: [run_experiment.sh](./run_experiment.sh). Use the bash shell to
execute the experiment using the wrapper script:

Example:

```shell
user@host:~/repo_path$ ./run_experiment.sh experiments/train_new_base.py
```

## Reproducing the results from the paper

1. Training LeNet-5 and KerasNet networks with linear units from scratch:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_lus base
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_lus ahaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_lus ahaf --dspu4
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_lus leaf --p24sl
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_lus leaf --p24sl --dspu4
   ```

2. Training LeNet-5 and KerasNet networks with linear units from scratch:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_bfs base
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_bfs ahaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_bfs ahaf --dspu4
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_bfs leaf --p24sl
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --opt adam --end_ep 100 --acts all_bfs leaf --p24sl --dspu4
   ```

3. On stability of LEAF-as-ReLU:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --end_ep 100 --acts ReLU --net KerasNet --ds CIFAR-10 \
             --opt adam leaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --end_ep 100 --acts ReLU --net KerasNet --ds CIFAR-10 \
             --opt adam leaf --p24sl
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --end_ep 100 --acts ReLU --net KerasNet --ds CIFAR-10 \
             --opt rmsprop ahaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --end_ep 100 --acts ReLU --net KerasNet --ds CIFAR-10 \
             --opt rmsprop leaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --end_ep 100 --acts ReLU --net KerasNet --ds CIFAR-10 \
             --opt rmsprop leaf --p24sl
   ```

   Add the `--wandb` parameter to log the training process to Weights and
   Biases. Weights and Biases provides visualization of the parameter values and
   the gradient values during training.

4. On the effect of synaptic weights initialization. Execute all commands below
   once per each of the seed values:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --seed 7823 --opt adam --ds CIFAR-10 base
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --seed 7823 --opt adam --ds CIFAR-10 ahaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --seed 7823 --opt adam --ds CIFAR-10 ahaf --dspu4
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --seed 7823 --opt adam --ds CIFAR-10 leaf --p24sl
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             --seed 7823 --opt adam --ds CIFAR-10 leaf --p24sl --dspu4
   ```

   Seed values to evaluate: 42, 100, 128, 1999, 7823.

## Visualization of experiment results

Use tools from the [post_experiment](./post_experiment) directory to visualize
training process, create the training result summary tables and visualize the
activation function form for LEAF/AHAF compared to the corresponding base
activations.
