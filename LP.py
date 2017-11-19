
import numpy as np
from gurobipy import *

def getQvalues(S,A,T,R,V,discount):
	Q = np.zeros((S,A))
	for s in range(0,S):
		for a in range(0,A):
			for n in range(0,S):
				Q[s][a] += T[s][a][n] * (R[s][a][n] + (discount * V[n]))
	return Q

def Primal(S, A, discount, horizon, epsilon, T, R):

	V = []


	model = Model("policy")
	P = [0 for i in range(0,S)]


	for s in range(0,S):
		for k in range(0,A):
				V.append(model.addVar(obj=1, vtype="C", name="y[%s]"%s)*model.addVar(obj=1, vtype="C", name="y[%s]"%s))

	model.update()


	Q2 = np.zeros((S,A))
	for s in range(S):
		for a in range(A):
			total = 0
			for s_n in range(S):
				total += T[s][a][s_n] * (R[s][a][s_n] + (discount * V[s_n]))
			model.addConstr(V[s] >= total,name="Q")


	model.update()
	model.optimize()

	#Retrieve the value function from the LP model
	finalV = []
	for s in range(S):
		finalV.append(V[s].x)

	#Derive optimal policy from Q*
	Q = getQvalues(S,A,T,R,finalV,discount)
	for s in range(0,S):
		P[s] = np.argmax(Q[s])

	return P,finalV
	
def Dual(S, A, discount, horizon, epsilon, T, R):

	P = [0 for i in range(0,S)]
	V = []
	x={}


	#Calculate the transition and rewards for each state / action
	c = np.zeros((S,A))
	for s in range(0,S):
		for a in range(0,A):
			for n in range(0,S):
				c[s][a] += R[s][a][n] * T[s][a][n]




	model = Model("policy")
	model.setObjective(GRB.MINIMIZE)




	y,x = {}, {}
	for i in range(S):
		for j in range(A):
			x[i,j] = model.addVar(obj=0)
			y[i,j] = model.addVar(obj=c[i,j], vtype="B", name="y[%s]"%j)

	model.setObjective(quicksum(x[i,k]*y[i,k]
                               for i in range(0,S) for k in range(0,A)))




	for j in range(S):
		coef = [1 for l in range(A)]
		var = [y[j,k] for k in range(A)]
		model.addConstr(LinExpr(coef,var) == 1+discount*quicksum(x[i,k]*T[j][k][i]
                               for i in range(0,S) for k in range(0,A)),name="Q")





	model.update()
	model.optimize()
	for i in range(S):
		print i
		q = []
		for j in range(A):
			q.append(y[i,j].x)
		print q
		P[i] = np.argmax(q)

	






	return P,V