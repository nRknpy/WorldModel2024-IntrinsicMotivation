from gymnasium_robotics.envs.maze.maps import *

RED = RR = 'red'
GREEN = GG = 'green'
BLUE = BB = 'blue'
NORMAL = NN = 'normal'

U_MAZE_COLOR = [
    [NN, NN, NN, NN, NN],
    [NN,  0,  0,  0, NN],
    [NN, RR, RR,  0, NN],
    [NN,  0,  0,  0, NN],
    [NN, NN, NN, NN, NN],
]

MEDIUM_MAZE_COLOR = [
    [NN, NN, NN, NN, RR, RR, RR, RR],
    [NN,  0,  0, NN, RR,  0,  0, RR],
    [NN,  0,  0, NN,  0,  0,  0, RR],
    [NN, NN,  0,  0,  0, RR, RR, RR],
    [BB,  0,  0, BB,  0,  0,  0, GG],
    [BB,  0, BB,  0,  0, GG,  0, GG],
    [BB,  0,  0,  0, GG,  0,  0, GG],
    [BB, BB, BB, BB, GG, GG, GG, GG],
]

LARGE_MAZE_COLOR = [
    [NN, NN, NN, NN, NN, NN, RR, RR, RR, RR, RR, RR],
    [NN,  0,  0,  0,  0, NN,  0,  0,  0,  0,  0, RR],
    [NN,  0, NN, NN,  0, NN,  0, RR,  0, RR,  0, RR],
    [NN,  0,  0,  0,  0,  0,  0, RR,  0,  0,  0, RR],
    [NN,  0, NN, NN, NN, NN,  0, RR, RR, RR,  0, RR],
    [BB,  0,  0, BB,  0, BB,  0,  0,  0,  0,  0, GG],
    [BB, BB,  0, BB,  0, BB,  0, GG,  0, GG, GG, GG],
    [BB,  0,  0, BB,  0,  0,  0, GG,  0,  0,  0, GG],
    [BB, BB, BB, BB, BB, BB, GG, GG, GG, GG, GG, GG],
]
