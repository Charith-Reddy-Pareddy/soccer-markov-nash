"""Certify the sparse deterministic Bellman fixed point with rational arithmetic.

The numerical solve supplies a candidate only; every equality below is checked
exactly with gamma=9/10 and the actual transition/reward function.
"""
import sys,json
from pathlib import Path
from math import log
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soccer_nash.game import SoccerGame,MOVE_ACTIONS
from soccer_nash.nash_q import NashQIteration
g=SoccerGame(move_order='deterministic');solver=NashQIteration(g,gamma=.9,mode='hybrid',tol=1e-11);r=solver.run_exact();gamma=Fraction(9,10)
v={}
for s,value in r.values.items():
 if value==0: v[s]=Fraction(0)
 else:
  rank=round(log(abs(value))/log(.9))
  assert 0 <= rank < len(r.values)
  power=gamma**rank
  assert abs(float(power)-abs(value))<1e-12
  v[s]=power if value>0 else -power
zeros=0
for state in g.states():
 matrix=[]
 for a in MOVE_ACTIONS:
  row=[]
  for b in MOVE_ACTIONS:
   _,ns,reward=g.transitions(state,a,b)[0]
   row.append(Fraction(reward[0])+ (0 if g.is_terminal(ns) else gamma*v[ns]))
  matrix.append(row)
 assert max(map(min,matrix)) == v[state] == min(max(row[j] for row in matrix) for j in range(4)),state
 zeros+=all(x==0 for row in matrix for x in row)
print(json.dumps({'states':len(v),'joint_actions':len(v)*16,'rational_gamma':'9/10','all_zero_Q':zeros,'certificate':'exact rational maximin = minimax = V at every state'}))
