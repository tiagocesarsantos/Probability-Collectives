import numpy as np
import random
import math

alpha = 1.0
agents = 3
mi = 3
T = 0.1
epsilon = 0.0001

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

domain = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
domain =np.array(domain, np.float)


# set of strategies
def define_set_strategies(agents, mi, domain):
  X = np.array([np.zeros(mi)])
  for i in range(0,agents-1):
    X = np.append(X, [np.zeros(mi)], 0)

  # randomize strategies for the agents
  for i in range(0, agents):
    bag_domain = domain.copy() # a bag to remove the numbers 
    for j in range(0, mi):
      rnd = random.randint(0, len(bag_domain)-1) # randomize a number
      X[i][j] = bag_domain[rnd]  # get a number from the bag
      bag_domain = np.delete(bag_domain, rnd, 0) # remove the element from the bag
    
        
  return X

# set of probabilities
def define_initial_probabilities(agents, mi):

  q = np.array([np.ones(mi)])
  for i in range(0,agents-1):
    q = np.append(q, [np.ones(mi)], 0)
    
  for i in range(0, agents):
    for j in range(0, mi):
      q[i][j] = q[i][j]/mi


  return q

# combined strategies
def define_combined_strategies(X, agents):
  Y = []
  for i in range(0, agents):
    Y.append(np.copy(X.T))    
  Y = np.array(Y)

  for i in range(0, agents):
    for j in range(0, agents):
      if i!=j:
        np.random.shuffle(Y[i][:,j])

  return Y


# compute objective function
def compute_objective_system(Y, agents, mi):
  G = []
  for i in range(0, agents):
    aux = []
    for r in range(0, mi):
      f = function_sphere(Y[i][r])
      aux.append(f)
    G.append(aux)

  return np.array(G)

  

def function_sphere(comb_strategy):
  sum_ = 0.0
  for i in range(0, len(comb_strategy)):
    sum_ = sum_ + math.pow(comb_strategy[i], 2.0)

  return sum_


def Si(i, mi, q): # i = agent, mi = strategy, q = probability
  si = 0.0 # sum
  for r in range(0, mi):
    si = si + (q[i][r] * math.log(q[i][r],2.0))
  return (-1.0*si)
  

def E(Y, G, q, i, r, mi): 
  E = G[i][r] * q[i][r]

  for j in range(0, agents):
    if i!=j:
      idx = np.where(Y[i][r][j] == X[j])[0][0]      
      E = E * q[j][idx] 

  return E


def update_q(Y, G, q, i, mi, T):

  E_ = np.zeros(mi)
  new_q = np.zeros(mi)
  E_sum = 0.0
  print("T= ", round(T, 3))
  for r in range(0, mi):
    E_[r] = E(Y, G, q, i, r, mi)
    E_sum = E_sum + E_[r]
  
  for r in range(0, mi):
    C = (E_[r]-E_sum)

    S = Si(i, mi, q)

    log = math.log(q[i][r],2.0)

    new_q[r] = q[i][r] - (alpha * q[i][r] * (((C)/T) + S + log))


    print ("i=",i, " r=", r, " G=", round(G[i][r],3), " q=", round(q[i][r],3), " C=", round(C,3), " E_=", round(E_[r],3), " E_sum=", round(E_sum,3), " S=", round(S, 3), " log=", round(log, 3), " newq=", round(new_q[r], 3))    
    
    if new_q[r] < 0.0:
      new_q[r] = 0.001

  return new_q

def update_step(Y, G, q, mi, T):
  new_q_array = np.array([np.ones(mi)])
  
  for i in range(0,agents-1):
    new_q_array = np.append(new_q_array, [np.ones(mi)], 0)
   
  for i in range(0, agents):
    new_q_array[i] = update_q(Y, G, q, i, mi, T)

  print new_q_array
  
  normalize(new_q_array)

  print new_q_array
  
  return new_q_array
  
def normalize(q):
  for i in range(0, len(q)):
    sumq = sum(q[i])
    for j in range(0, len(q[i])):
      q[i][j] = q[i][j]/sumq

def favorable_strategy(X, q):
  #np.argmax(q[0])
  Y_fav = []
  Y_fav_idx = []
  
  for i in range(0, agents):
    Y_fav.append(X[i][np.argmax(q[i])])
    Y_fav_idx.append(np.argmax(q[i]))

  return np.array(Y_fav), Y_fav_idx
  
def favorable_system_objective(Y_fav):
  
  return function_sphere(Y_fav)


last_q = []
def start(T, epsilon):
  global q,   last_q
  last_q = np.copy(q)
  k = 0
  while (1):
    q = update_step(Y,G,q,mi,T)
    #Y_fav, Y_fav_idx = favorable_strategy(X, q)
    #G_fav = favorable_system_objective(Y_fav)
    
    
    print "last_q - q", np.square(q) - np.square(last_q), " lastq",np.abs(np.mean(np.square(q) - np.square(last_q)))
        
    
    k=k+1
    print "k=", k, np.abs(np.mean(np.square(q) - np.square(last_q)))
    
    if np.abs(np.mean(np.square(q) - np.square(last_q))) < epsilon:
      break

    last_q = np.copy(q)

  Y_fav, Y_fav_idx = favorable_strategy(X, q)
  G_fav = favorable_system_objective(Y_fav)
  print "G_fav = ", G_fav



X = define_set_strategies(agents, mi, domain)
q = define_initial_probabilities(agents, mi)
Y = define_combined_strategies(X, agents)
G = compute_objective_system(Y, agents, mi)












