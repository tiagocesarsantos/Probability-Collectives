import numpy as np
import math

alpha = 1.0
agents = 3
mi = 3
T = 0.1
epsilon = 0.0001

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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

def update_step(Y, G, q, mi):
  global T
  new_q_array = np.array([np.ones(mi)])
  
  for i in range(0,agents-1):
    new_q_array = np.append(new_q_array, [np.ones(mi)], 0)
   
  for i in range(0, agents):
      new_q_array[i] = update_q(Y, G, q, i, mi, T)

  print new_q_array
  
  normalize(new_q_array)
  #T = T-0.01

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


def start():
  global q
  for k in range(0, 1000):
    q = update_step(Y,G,q,mi)
    Y_fav, Y_fav_idx = favorable_strategy(X, q)
    G_fav = favorable_system_objective(Y_fav)
    print "G_fav = ", G_fav
    
    

"""
X= np.array([[ 0., -3.,  5.],
       [ 0.,  4.,  5.],
       [-1., -3.,  2.]])
Y= np.array([[[ 0.,  0., -3.],
        [-3.,  4., -1.],
        [ 5.,  5.,  2.]],

       [[ 5.,  0.,  2.],
        [ 0.,  4., -1.],
        [-3.,  5., -3.]],

       [[ 5.,  5., -1.],
        [-3.,  0., -3.],
        [ 0.,  4.,  2.]]])
G = np.array([[ 9., 26., 54.],
       [29., 17., 43.],
       [51., 18., 20.]])

q = np.array([[0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333]])
"""       

X= np.array([[ 0., -3.,  5.],
       [ 0.,  4.,  5.],
       [-1., -3.,  2.]])
Y= np.array([
        [[ 0.,  5., -3.],
        [-3.,  4., 2.],
        [ 5.,  0.,  -1.]],

       [[ 5.,  0.,  2.],
        [ 0.,  4., -1.],
        [-3.,  5., -3.]],

       [[ 5.,  5., -1.],
        [0.,  4., -3.],
        [ -3.,  0.,  2.]]])
G = np.array([[ 34., 29., 26.],
       [29., 17., 43.],
       [51., 25., 13.]])

q = np.array([[0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333]])

             
"""
X= np.array([[ 0., -3.,  5.],
       [ 0.,  4.,  5.]])

Y= np.array([[[ 0.,  0.],
        [-3.,  4.],
        [ 5.,  5.]],

       [[ -3.,  0.],
        [ 0.,  4.],
        [5.,  5.]]])

        
G = np.array([[ 0., 25., 50.],
       [9., 16., 50.]])

q = np.array([[0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333]])
       
"""

"""
X= np.array([[ 0., -3.,  5.],
       [ 0.,  4.,  5.]])

Y= np.array([[[ 0.,  4.],
        [-3.,  5.],
        [ 5.,  0.]],

       [[ 0.,  0.],
        [ -3.,  4.],
        [5.,  5.]]])

        
G = np.array([[ 16., 34., 25.],
       [0., 25., 50.]])

q = np.array([[0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.33333333, 0.33333333]])
       
"""


















