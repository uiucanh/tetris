# Tetris - Genetic Algorithm
This repo contains the implementation of an agent for the original Tetris (GameBoy)
using genetics aglorithm.

Medium article: 

# Installation
Using `pip`

```
pip install -r requirements.txt
```

Follows installation for PyBoy at https://github.com/Baekalfen/PyBoy#installation

# Training
To train with the approach in the article, run `python tetris.py` make sure that ROM is available in the directory and named 
`tetris_1.1.gb`.

There's also an implementation of [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) 
which you can find in `tetris_neat.py`

# Play
Inside `models`, there's a file `best.pkl` which contains the best model obtained
after 10 epochs, run `python play.py` to get an average 10 runs scores of the model

To play the games with the model from NEAT, use `play_neat.py`
