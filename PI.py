import numpy as np

def policyEvaluation(S, A, discount, horizon, epsilon, T, R,P):
    V = np.zeros(S)
    for timestep in range(0,horizon):
    	oldV = np.copy(V)
    	#Calculate the Q-values for each state,action,next
    	Q = getPolicyQ(S,A,T,R,V,discount,P)
    	for state in range(0,S):
    		v = V[state]
    		V[state] = sum(Q[state])
    	delta = np.max(np.fabs(oldV - V))
    	if (delta <= epsilon):
    		break
    return V


def howardImprovement(S,A,P,Q):
	policyStable = True
	improvedStates = np.zeros(S)
	for state in range(0,S):
		temp = P[state]
		P[state] = np.argmax(Q[state])
		if temp != P[state]:
			policyStable = False
	return policyStable

def simpleImprovement(S,A,P,Q):
	policyStable = True
	improvedStates = np.zeros(S)
	for state in range(0,S):
		temp = P[state]
		P[state] = np.argmax(Q[state])
		if temp != P[state]:
			policyStable = False
			break
	return policyStable



def getPolicyQ(S,A,T,R,V,discount,P):
	Q = np.zeros((S,A))
	for s in range(0,S):
		for a in range(0,A):
			for n in range(0,S):
				actionPolicy = int(P[s])
				if a == actionPolicy:
					Q[s][a] += T[s][actionPolicy][n] * (R[s][actionPolicy][n] + (discount * V[n]))
	return Q

def getQ(S,A,T,R,V,discount):
	Q = np.zeros((S,A))
	for s in range(0,S):
		for a in range(0,A):
			for n in range(0,S):
				Q[s][a] += T[s][a][n] * (R[s][a][n] + (discount * V[n]))
	return Q

def HowardPolicyIteration(S, A, discount, horizon, epsilon, T, R):
	 # Evaluate the current policy
	policyStable = False 
	timestep = 0
	P = [0 for i in range(0,S)]
	timestep=0
	while not policyStable:
		V = policyEvaluation(S,A,discount,horizon,epsilon,T,R,P)
		Q = getQ(S,A,T,R,V,discount)
		policyStable = howardImprovement(S,A,P,Q)
		timestep+=1
	print "HowardPolicyIteration converged at timestep ",timestep
	return P, V

def SimplePolicyIteration(S, A, discount, horizon, epsilon, T, R):
	 # Evaluate the current policy
	policyStable = False 
	timestep = 0
	P = [0 for i in range(0,S)]
	timestep=0
	while not policyStable:
		V = policyEvaluation(S,A,discount,horizon,epsilon,T,R,P)
		Q = getQ(S,A,T,R,V,discount)
		policyStable = simpleImprovement(S,A,P,Q)
		timestep+=1
	print "SimplePolicyIteration converged at timestep ",timestep
	return P, V

