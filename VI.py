import numpy as np


def valueIteration(S, A, discount, horizon, epsilon, T, R):
    V = np.zeros(S)
    P = [0 for i in range(0,S)]
  

    for timestep in range(0,horizon):
    	oldV = np.copy(V)

    	#Calculate the Q-values for each state,action,next
    	Q = getQvalues(S,A,T,R,V,discount)
    	
    	#Select the maximum Q-values for each state = bellman optimality
    	for state in range(0,S):
    		bestAction = np.max(Q[state])
    		bestIndex = np.argmax(Q[state])
    		V[state] = bestAction
    		P[state] = bestIndex

    	delta = np.max(np.fabs(oldV - V))
    	if (delta <= epsilon):
    		print "Value iteration converged at timestep ",timestep
    		break
    return P, V

def getQvalues(S,A,T,R,V,discount):
	Q = np.zeros((S,A))
	for s in range(0,S):
		for a in range(0,A):
			for n in range(0,S):
				Q[s][a] += T[s][a][n] * (R[s][a][n] + (discount * V[n]))
	return Q

def initializeRewardsQ(S,A,T,R,V,discount):
    Q = np.zeros((S,A))
    for s in range(0,S):
        for a in range(0,A):
            for n in range(0,S):
                Q[s][a] += T[s][a][n] * R[s][a][n]
    return Q


def smallBackups(S,A,T,R,V,discount,d,Q):
    for s in range(0,S):
        for a in range(0,A):
            for n in range(0,S):
                diff = V[s] - d[n][a]
                d[n][a] = V[n]
                Q[s][a] = Q[s][a] +  discount *  (T[s][a][n] * diff)
    return Q

def valueIterationSmallBackup(S, A, discount, horizon, epsilon, T, R):
    V = np.zeros(S)
    P = [0 for i in range(0,S)]

    OldQ = np.zeros((S,A))
    d = np.zeros((S,A))
    #Initialize the Q-values equal to the rewards
    Q = initializeRewardsQ(S,A,T,R,V,discount)



    for timestep in range(0,1):
        oldV = np.copy(V)

        #Initialize the Q-values equal to the rewards
        Q = initializeRewardsQ(S,A,T,R,V,discount)
 
      
        #Select the maximum Q-values for each state = bellman optimality
        for state in range(0,S):
            bestAction = np.max(Q[state])
            bestIndex = np.argmax(Q[state])
            P[state] = bestIndex

        

       
        Q = smallBackups(S,A,T,R,V,discount,d,Q)
        
   
        


        delta = np.max(np.fabs(oldV - V))
        if (delta <= epsilon):
            print "Value iteration converged at timestep ",timestep
            break
    return P, V




