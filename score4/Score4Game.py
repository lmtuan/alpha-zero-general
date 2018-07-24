from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
#import .Score4Symmetries as Symm
from . import Score4Symmetries as Symm
import numpy as np

class Score4Board():
    def __init__(self, white_pieces = 0, black_pieces = 0):
        "Set up initial board configuration."
        # Create the empty bitboard for each player
        self.pieces = [white_pieces, black_pieces]

class Score4Game(Game):
    BOARD_SIZE = 4
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

    def __init__(self):
        pass

    def getInitBoard(self):
        # return initial board
        b = Score4Board()
        return b

    def getBoardSize(self):
        return (self.BOARD_SIZE, self.BOARD_SIZE, self.BOARD_SIZE)

    def getActionSize(self):
        # return number of actions (columns)
        return self.ACTION_SIZE

    def getNextState(self, board, player, action):
        # an action is 0 to 15, indicating which column the new piece is dropped
        playerInd = int(player < 0) # 0/1 for white/black
        newBoard = Score4Board()
        newBoard.pieces = board.pieces.copy()

        allPieces = board.pieces[0] | board.pieces[1]
        for pos in range(action, 64, self.ACTION_SIZE):
            if (allPieces >> pos) & 1 == 0: # empty position
                newBoard.pieces[playerInd] |= 1 << pos
                break
        return (newBoard, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        lastLayer = (board.pieces[0] | board.pieces[1]) >> 48
        validMoves = np.zeros(self.getActionSize())

        for i in range(0, self.getActionSize()):
            validMoves[i] = (lastLayer >> i & 1) == 0
        return validMoves

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        if (board.pieces[0] | board.pieces[1]) == 0xFFFFFFFFFFFFFFFF:
            return 1e-4
        elif self._getGameEnded(board.pieces[0]): # white win
            return 1 if player == 1 else -1
        elif self._getGameEnded(board.pieces[1]): # black win
            return 1 if player == -1 else -1
        else:
            return 0

    def _getGameEnded(self, brd):
        '''brd is a 64-bit bitboard
        Return:
            a non-zero number if the bitboard contains 4 in a row
            zero if the bitboard doesn't have 4 in a row
        '''
        
        # win in a straight line
        return (brd & brd >> 1  & brd >> 2  & brd >> 3  & 0x1111111111111111) \
        | (brd & brd >> 4  & brd >> 8  & brd >> 12 & 0x000f000f000f000f) \
        | (brd & brd >> 16 & brd >> 32 & brd >> 48 & 0x000000000000ffff) \
        | (brd & brd >> 5  & brd >> 10 & brd >> 15 & 0x0001000100010001) \
        | (brd & brd >> 3  & brd >> 6  & brd >> 9  & 0x0008000800080008) \
        | (brd & brd >> 17 & brd >> 34 & brd >> 51 & 0x0000000000001111) \
        | (brd & brd >> 15 & brd >> 30 & brd >> 45 & 0x0000000000008888) \
        | (brd & brd >> 20 & brd >> 40 & brd >> 60 & 0x000000000000000f) \
        | (brd & brd >> 12 & brd >> 24 & brd >> 36 & 0x000000000000f000) \
        | (brd & brd >> 21 & brd >> 42 & brd >> 63 & 0x0000000000000001) \
        | (brd & brd >> 11 & brd >> 22 & brd >> 33 & 0x0000000000008000) \
        | (brd & brd >> 19 & brd >> 38 & brd >> 57 & 0x0000000000000008) \
        | (brd & brd >> 13 & brd >> 26 & brd >> 39 & 0x0000000000001000)

    def getCanonicalForm(self, board, player):
        # return board if player == 1, or a flipped board if player == -1
        newBoard = Score4Board()
        if player == 1:
            newBoard.pieces[0] = board.pieces[0]
            newBoard.pieces[1] = board.pieces[1]
        elif player == -1:
            newBoard.pieces[0] = board.pieces[1]
            newBoard.pieces[1] = board.pieces[0]
        else:
            raise ValueError('Unexpected player: ', player)
        return newBoard

    # modified
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.getActionSize())
        
        # rotate 0, 90, 180, 270 degree
        syms = [(board, pi)]
        for i in range(0, 3):
            board, pi = self._rotate90(board, pi)
            syms.append( (board, pi) )
        # mirrored rotate 0, 90, 180, 270 degree
        board, pi = self._mirror(board, pi)
        syms.append( (board, pi) )
        for i in range(0, 3):
            board, pi = self._rotate90(board, pi)
            syms.append( (board, pi) )
        return syms

    def _rotate90(self, board, pi):
        """Rotate a board object and pi array 90 degree
        Return:
            a tuple of rotated board object and pi
        """
        newBoard = Score4Board(
            Symm.rotateBoard90(board.pieces[0]),
            Symm.rotateBoard90(board.pieces[1]))
        newPi = Symm.rotatePi90(pi)
        return newBoard, newPi

    def _mirror(self, board, pi):
        """Mirror a board object
        Return
            a a tuple of mirrored board object and pi
        """
        newBoard = Score4Board(
            Symm.mirrorBoard(board.pieces[0]),
            Symm.mirrorBoard(board.pieces[1]))
        newPi = Symm.mirrorPi(pi)
        return newBoard, newPi

    def stringRepresentation(self, board):
        s = ""
        for i in range(0, 64, 8):
            s += chr(board.pieces[0] >> i & 0xFF)
            s += chr(board.pieces[1] >> i & 0xFF)
        return format(board.pieces[0], 'x') + format(board.pieces[1], 'x')


def display(board):
    H_SEP = "-----------------"
    X_PIECE = " x "
    O_PIECE = " o "
    EMPTY   = "   "
    # print 3 upper layers
    for h in range(3, 0, -1):
        for r in range(3, -1, -1):
            offset = 2 * r + 1
            print( (offset+1) * ' ', end="")
            print(H_SEP)
            print(offset * ' ', end="/")
            for c in range(0, 4):
                if   board.pieces[0] >> (h*16+r*4+c) & 1:
                    print(X_PIECE, end="/")
                elif board.pieces[1] >> (h*16+r*4+c) & 1:
                    print(O_PIECE, end="/")
                else:
                    print(EMPTY, end="/")
            print()
        print(H_SEP)

    # print last layer with enumerated column
    for r in range(3, -1, -1):
        offset = 2 * r + 1
        print( (offset+1) * ' ', end="")
        print(H_SEP, end="")
        print(10 * ' ', end="")
        print(H_SEP)

        print(offset * ' ', end="/")
        # a row of last layer
        for c in range(0, 4):
            if   board.pieces[0] >> (r*4+c) & 1:
                print(X_PIECE, end="/")
            elif board.pieces[1] >> (r*4+c) & 1:
                print(O_PIECE, end="/")
            else:
                print(EMPTY, end="/")
        # a row of enumerated column
        print(10 * ' ', end="/")
        for c in range(0, 4):
            print("{0:^3}".format(str(r*4+c)), end="/")
        print()
    print("{0}{1}{0}".format(H_SEP, 10 * ' '))