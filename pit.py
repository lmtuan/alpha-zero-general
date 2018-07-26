import Arena
from MCTS import MCTS
from score4.Score4Game import Score4Game, display
from score4.Score4Players import *
from score4.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def displayBoard(b):
    pass

g = Score4Game()

# all players
rp = RandomPlayer(g).play
gp = GreedyScore4Player(g).play
hp = HumanScore4Player(g).play
lp = LookAheadPlayer(g, 4).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/score4_2_always_accept','checkpoint_3.pth.tar')
args1 = dotdict({'numMCTSSims': 200, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g)
n2.load_checkpoint('./temp/score4_1','best.pth.tar')
args2 = dotdict({'numMCTSSims': 200, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
