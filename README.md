# Tetris - Genetic Algorithm
This repo contains the implementation of an agent for the original Tetris (GameBoy)
using genetics aglorithm.

[Medium article](https://medium.com/@bdanh96/beating-the-world-record-in-tetris-gb-with-genetics-algorithm-6c0b2f5ace9b) 

# Installation
Using `pip`

```
pip install -r requirements.txt
```

Follows installation for PyBoy at https://github.com/Baekalfen/PyBoy#installation

# Training
To train with the approach in the article, run `python tetris.py` make sure that ROM is available in the directory and named 
`tetris.gb`.

```
usage: Tetris training [-h] [-e EPOCHS] [-r RUNS] [-p POPSIZE] [-m MAXSCORE] [-n NUMWORKERS] [-s]

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        How many epochs to train
  -r RUNS, --runs RUNS  Tetris games played per child
  -p POPSIZE, --popsize POPSIZE
                        Size of population
  -m MAXSCORE, --maxscore MAXSCORE
                        Maximum score in one run, game is stopped if reached
  -n NUMWORKERS, --numworkers NUMWORKERS
                        Number of workers (pyboys) to spawn
  -s, --show            Do not run pyboys in headless mode, render each one in window
```

There's also an implementation of [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) 
which you can find in `tetris_neat.py`

```
usage: Tetris neat model trainer [-h] [-c CHECKPOINT_PATH] [-e EPOCHS] [-n NUM_WORKERS] [-a ACTION_LIMIT] [-s] [-d]

options:
  -h, --help            show this help message and exit
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        Path to checkpoint to continue simulation
  -e EPOCHS, --epochs EPOCHS
                        Epochs to run simulation
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers to spawn for simulation
  -a ACTION_LIMIT, --action_limit ACTION_LIMIT
                        Maximum number of moves to make in simulation
  -s, --show            Do not run pyboys in headless mode, render each one in window
  -d, --draw            Draw best network after simulation
```

To configure your NEAT experiment use `config\config-feedforward.txt`, check the [docs](https://neat-python.readthedocs.io/en/latest/config_file.html)
for details.

# Play
Inside `models`, there's a file `best.pkl` which contains the best model obtained
after 10 epochs, run `python play.py models/best.pkl` to check it out:

```
usage: Tetris model player [-h] [-r RUNS] model_path

positional arguments:
  model_path            Path to model

options:
  -h, --help            show this help message and exit
  -r RUNS, --runs RUNS  Tetris games played
```

To play the games with the model from NEAT, use `play_neat.py`:

```
usage: Tetris neat model player [-h] [-r RUNS] [-d] model_path

positional arguments:
  model_path            Path to neat model

options:
  -h, --help            show this help message and exit
  -r RUNS, --runs RUNS  Tetris games played
  -d, --draw            Draw network after playing
```
