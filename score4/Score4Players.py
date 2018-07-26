import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanScore4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        while True:
            x = int(input())
            if valid[x]:
                break
            else:
                print('Invalid')

        return x


class GreedyScore4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self._getScore(nextBoard)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

    def _getScore(self, board):
        return self.game.getGameEnded(board, 1)


class LookAheadPlayer():
    def __init__(self, game, depth):
        self.game = game
        self.depth = depth

    def play(self, board):
        val, move = self.lookAhead(board, self.depth, -9999, 9999, 1)
        print('LookeAheadPlayer({depth}): val={0} move={1}'.format(val, move, depth=self.depth))
        return move

    def lookAhead(self, board, depth, alpha, beta, player):
        if depth == 0 or self.game.getGameEnded(board, player) != 0:
            return self.game.getGameEnded(board, player), -1

        valids = self.game.getValidMoves(board, player)
        bestVal = -9999
        bestMove = -1
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, player, a)
            val = -self.lookAhead(nextBoard, depth-1, -beta, -alpha, -player)[0]
            if val > bestVal:
                bestVal = val
                bestMove = a
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return bestVal, bestMove