from score4.Score4Game import *

g = Score4Game()
b = g.getInitBoard()
# display(b)
#print(g.getValidMoves(b, 1))
#print(g.getGameEnded(b, 1))

plr = 1
b, plr = g.getNextState(b, plr, 15)
b, plr = g.getNextState(b, plr, 15)
b, plr = g.getNextState(b, plr, 15)
b, plr = g.getNextState(b, plr, 15)
b, plr = g.getNextState(b, plr, 14)
b, plr = g.getNextState(b, plr, 14)
b, plr = g.getNextState(b, plr, 14)
b, plr = g.getNextState(b, plr, 14)
b, plr = g.getNextState(b, plr, 13)
b, plr = g.getNextState(b, plr, 13)
b, plr = g.getNextState(b, plr, 12)


pi = [i for i in range(16)]
syms = g.getSymmetries(b, pi)
for b2, pi2 in syms:
  print('=============================================================')
  display(b2)
  #print(pi2)
  #print(g.getValidMoves(b2, 1))
  #print(g.stringRepresentation(b2))
  print(g.getGameEnded(b2, 1))
  print(g.getGameEnded(b2, -1))
