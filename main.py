from Coach import Coach
from score4.Score4Game import Score4Game as Game
from score4.tensorflow.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 256,
    'tempThreshold': 25,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 40,
    'cpuct': 3.0,

    'checkpoint': './temp/score4_2_always_accept',
    'load_model': True,
    'load_folder_file': ('./temp/score4_2_always_accept','best.pth.tar'),
    'numItersForTrainExamplesHistory': 15,

})

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
