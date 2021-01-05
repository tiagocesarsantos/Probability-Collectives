import numpy as np
import random
import math
import matplotlib.pyplot as plt


agents = 5
mi = 10
alpha = 1.0
T = 1.0 #0.001
epsilon = 0.0001
k_final = 100

#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

domain = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
domain =np.array(domain, np.float)

_min_domain = -5.12
_max_domain = 5.12

# main function
def function_sphere(comb_strategy):
  sum_ = 0.0
  for i in range(0, len(comb_strategy)):
    sum_ = sum_ + math.pow(comb_strategy[i], 2.0)

  return sum_


# get value from domain, just a uniform random
def get_value_from_domain(_min_domain, _max_domain):
  return random.uniform(_min_domain, _max_domain)

# set of strategies
def define_set_strategies(agents, mi, domain):
  # create array
  X = np.array([np.zeros(mi)])
  for i in range(0,agents-1):
    X = np.append(X, [np.zeros(mi)], 0)

  # randomize strategies for the agents  
  for i in range(0, agents):
    for j in range(0, mi):
      X[i][j] = get_value_from_domain(_min_domain, _max_domain)

  return X

# set of probabilities
def define_initial_probabilities(agents, mi):

  # create array
  q = np.array([np.ones(mi)])
  for i in range(0,agents-1):
    q = np.append(q, [np.ones(mi)], 0)
    
  # initial guess is 1/mi for every strategy
  for i in range(0, agents):
    for j in range(0, mi):
      q[i][j] = q[i][j]/mi

  return q

# combined strategies
def define_combined_strategies(X, agents):
  # create array
  Y = []
  for i in range(0, agents):
    Y.append(np.copy(X.T))    
  Y = np.array(Y)

  # shuffle combined strategies Y #TODO show the formula in pdf
  for i in range(0, agents):
    for j in range(0, agents):
      if i!=j:
        np.random.shuffle(Y[i][:,j])

  return Y


# compute objective function
#TODO calculates G, 
def compute_objective_system(Y, agents, mi):
  G = []
  for i in range(0, agents):
    aux = []
    for r in range(0, mi):
      f = function_sphere(Y[i][r])
      aux.append(f)
    G.append(aux)

  return np.array(G)

  


# Si function
def Si(i, mi, q): # i = agent, mi = strategy, q = probability
  si = 0.0 # sum
  for r in range(0, mi):
    si = si + (q[i][r] * math.log(q[i][r],2.0))
  return (-1.0*si)
  
# Expected utility function
def E(Y, G, q, i, r, mi): 
  E = 0.0
  E = G[i][r] * q[i][r]

  for j in range(0, agents):
    if i!=j:
      idx = np.where(Y[i][r][j] == X[j])[0][0]      
      E = E * q[j][idx] 

  return E

# function to update the probability distribution
def update_q(Y, G, q, i, mi, T):

  E_ = np.zeros(mi)
  new_q = np.zeros(mi)
  E_sum = 0.0
  #print("T= ", round(T, 3))
  for r in range(0, mi):
    E_[r] = E(Y, G, q, i, r, mi)
    E_sum = E_sum + E_[r]
  
  for r in range(0, mi):
    C = (E_[r]-E_sum)

    S = Si(i, mi, q)

    log = math.log(q[i][r],2.0)

    new_q[r] = q[i][r] - (alpha * q[i][r] * (((C)/T) + S + log))


    #print ("i=",i, " r=", r, " G=", round(G[i][r],3), " q=", round(q[i][r],3), " C=", round(C,3), " E_=", round(E_[r],3), " E_sum=", round(E_sum,3), " S=", round(S, 3), " log=", round(log, 3), " newq=", round(new_q[r], 3))    
    
    if new_q[r] < 0.0001:
      new_q[r] = 0.0001

  return new_q

def update_step(Y, G, q, mi, T):
  new_q_array = np.array([np.ones(mi)])
  
  for i in range(0,agents-1):
    new_q_array = np.append(new_q_array, [np.ones(mi)], 0)
   
  for i in range(0, agents):
    new_q_array[i] = update_q(Y, G, q, i, mi, T)

  #print new_q_array
  
  normalize(new_q_array)

  #print new_q_array
  
  return new_q_array
  
# normalize the probability distribution
def normalize(q):
  for i in range(0, len(q)):
    sumq = sum(q[i])
    for j in range(0, len(q[i])):
      q[i][j] = q[i][j]/sumq

# get favorable strategy
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
best_Y_fav = []
best_Y_fav_idx = []
best_G_fav = []

last_Y_fav = []
last_Y_fav_idx = []
last_G_fav = []

def start(T, epsilon):
  step = []
  G_step = []
  global q,   last_q
  last_q = np.copy(q)
  k = 0
  while (1):
    q = update_step(Y,G,q,mi,T)
    #Y_fav, Y_fav_idx = favorable_strategy(X, q)
    #G_fav = favorable_system_objective(Y_fav)
    
    
    #print "last_q - q", np.square(q) - np.square(last_q), " lastq",np.abs(np.mean(np.square(q) - np.square(last_q)))      
    k=k+1
    print ("k = ", k)
    #print "k=", k, np.abs(np.mean(np.square(q) - np.square(last_q)))
    
    # TODO break rule if epsilon
    #if np.abs(np.mean(np.square(q) - np.square(last_q))) < epsilon:
    #  break
    # TODO break rule k < k_final
    if (k > k_final):
      break

    Y_fav, Y_fav_idx = favorable_strategy(X, q)
    G_fav = favorable_system_objective(Y_fav)
    G_step.append(G_fav)
    step.append(k)
    print "G_fav = ", G_fav
    print "Y_fav_idx = ",Y_fav_idx

    last_q = np.copy(q)

  G_step.append(G_fav)
  step.append(k)
  G_step = np.array(G_step)
    
  Y_fav, Y_fav_idx = favorable_strategy(X, q)
  G_fav = favorable_system_objective(Y_fav)
  print "G_fav = ", G_fav
  print "Y_fav_idx = ",Y_fav_idx
  print "Y_fav = ", Y_fav


  plt.plot(step, G_step)
  plt.plot(step, np.ones(len(step))*np.min(G), color='r')
  plt.show()

  return G_fav, Y_fav_idx, Y_fav

X = define_set_strategies(agents, mi, domain)
q = define_initial_probabilities(agents, mi)
Y = define_combined_strategies(X, agents)
G = compute_objective_system(Y, agents, mi)



# execute
T= 1.0
best_G_fav, best_Y_fav_idx, best_Y_fav = start(T, epsilon)


q = define_initial_probabilities(agents, mi)
T= 0.1
last_G_fav, last_Y_fav_idx, last_Y_fav = start(T, epsilon)
if last_G_fav < best_G_fav:
  best_G_fav, best_Y_fav_idx, best_Y_fav = last_G_fav, last_Y_fav_idx, last_Y_fav 

q = define_initial_probabilities(agents, mi)
T= 0.01
last_G_fav, last_Y_fav_idx, last_Y_fav = start(T, epsilon)
if last_G_fav < best_G_fav:
  best_G_fav, best_Y_fav_idx, best_Y_fav = last_G_fav, last_Y_fav_idx, last_Y_fav

q = define_initial_probabilities(agents, mi)
T= 0.001
last_G_fav, last_Y_fav_idx, last_Y_fav = start(T, epsilon)
if last_G_fav < best_G_fav:
  best_G_fav, best_Y_fav_idx, best_Y_fav = last_G_fav, last_Y_fav_idx, last_Y_fav

q = define_initial_probabilities(agents, mi)
T= 0.0001
last_G_fav, last_Y_fav_idx, last_Y_fav = start(T, epsilon)
if last_G_fav < best_G_fav:
  best_G_fav, best_Y_fav_idx, best_Y_fav = last_G_fav, last_Y_fav_idx, last_Y_fav

q = define_initial_probabilities(agents, mi)
T= 0.00001
last_G_fav, last_Y_fav_idx, last_Y_fav = start(T, epsilon)
if last_G_fav < best_G_fav:
  best_G_fav, best_Y_fav_idx, best_Y_fav = last_G_fav, last_Y_fav_idx, last_Y_fav





"""

execfile('pc3.py')
q = define_initial_probabilities(agents, mi)
T= 1.0
alpha = 1.0
start(T, epsilon)


q = define_initial_probabilities(agents, mi)
T= 0.1
start(T, epsilon)


q = define_initial_probabilities(agents, mi)
T= 0.01
start(T, epsilon)

q = define_initial_probabilities(agents, mi)
T= 0.001
start(T, epsilon)

q = define_initial_probabilities(agents, mi)
T= 0.0001
start(T, epsilon)

q = define_initial_probabilities(agents, mi)
T= 0.00001
start(T, epsilon)




execfile('pc3.py')

X=np.array([[-2.000, 3.000, 0.000],
       [-1.000, -2.000, 5.000],
       [0.000, -1.000, 4.000]])

# normal 2 good values       
Y=np.array([[[-2.000, -1.000, 4.000],
        [3.000, 5.000, 0.000],
        [0.000, -2.000, -1.000]],

       [[0.000, -1.000, 4.000],
        [3.000, -2.000, 0.000],
        [-2.000, 5.000, -1.000]],

       [[3.000, 5.000, 0.000],
        [0.000, -1.000, -1.000],
        [-2.000, -2.000, 4.000]]])

# 1 best value
Y=np.array([[[-2.000, -2.000, 4.000],
        [3.000, 5.000, -1.000],
        [0.000, -1.000, 0.000]],

       [[0.000, -1.000, -1.000],
        [3.000, -2.000, 0.000],
        [-2.000, 5.000, 4.000]],

       [[3.000, 5.000, 0.000],
        [0.000, -1.000, -1.000],
        [-2.000, -2.000, 4.000]]])

# 2 best value
Y=np.array([[[-2.000, -2.000, 4.000],
        [3.000, 5.000, -1.000],
        [0.000, -1.000, 0.000]],

       [[0.000, -1.000, -1.000],
        [3.000, -2.000, 0.000],
        [-2.000, 5.000, 4.000]],

       [[0.000, -1.000, 0.000],
        [3.000, 5.000, -1.000],
        [-2.000, -2.000, 4.000]]])

                
Y=np.array([[[-2.000, -1.000, 4.000],
        [3.000, -2.000, -1.000],
        [0.000, 5.000, 0.000]],

       [[3.000, -1.000, -1.000],
        [0.000, -2.000, 4.000],
        [-2.000, 5.000, 0.000]],

       [[3.000, -1.000, 0.000],
        [0.000, 5.000, -1.000],
        [-2.000, -2.000, 4.000]]])

        
G=np.array([[21.000, 34.000, 5.000],
       [17.000, 13.000, 30.000],
       [34.000, 2.000, 24.000]])

q = np.array([[0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333]])



q = define_initial_probabilities(agents, mi)
G = compute_objective_system(Y, agents, mi)


q = np.array([[0.4, 0.3, 0.3],
       [0.5, 0.3, 0.2],
       [0.3, 0.6, 0.1]])     
       
       
q = update_step(Y,G,q,mi,T)
"""







