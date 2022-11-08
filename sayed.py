import numpy as np
from math import sqrt

# define network
N = 5;
A = np.random.randint(0,5,size=(N,N)) # A(l,k): weight of edge directed from l to k
A = A / np.sum(A,axis=0)

# define actual model
M = 20;
w = np.random.randint(low=0,high=10,size=(M,1))
# make w sparse
mask = np.random.randint(low=0,high=2,size=(M,1))
w = w * mask

#  collect data from every agent
#  every agent probes the model
var_v = 0.1
Nt = 100
U = np.zeros(shape=(N,Nt)) # for recording agent inputs
D = np.zeros(shape=(N,Nt)) # for recording agent outputs
for k in range(0,N):
    uk = np.random.randn(1,Nt) # generate input according to standard gaussian
    dk = np.zeros(shape=(1,Nt)) 
    vk = sqrt(var_v)*np.random.randn(1,Nt)
    for i in range(M,Nt):
        uki = np.flip(uk[:,i-M+1:i+1],axis=1)
        dk[:,i] = uki @ w + vk[:,i] # convolve and add white gaussian noise
    U[k,:] = uk
    D[k,:] = dk

# diffusion-adaptation
mu = 0.1
w_est_prev = np.random.randn(N,M)
w_est = np.zeros(shape=(N,M)) # hold estimates

for i in range(M,Nt):
    for k in range(0,N):
        phi = np.zeros(shape=(M,1));
        for l in range(0,N):
            phi = phi + A[l,k] * np.expand_dims(w_est_prev[l,:],-1)
        uki = np.expand_dims(np.flip(U[k,i-M+1:i+1]),0)
        w_est_prev[k,:] = w_est[k,:]
        w_est[k,:] = np.squeeze(phi + mu* uki.transpose() @ (D[k,i] - uki @ phi))

# diffusion-adaptation with sparsity
mu = 0.1
lamda = 1e-3 # regularization parameter
w_est_prev = np.random.randn(N,M)
w_est = np.zeros(shape=(N,M)) # hold estimates

for i in range(M,Nt):
    for k in range(0,N):
        phi = np.zeros(shape=(M,1));
        for l in range(0,N):
            phi = phi + A[l,k] * np.expand_dims(w_est_prev[l,:],-1)
        uki = np.expand_dims(np.flip(U[k,i-M+1:i+1]),0)
        reg_term = np.diag(1/(np.abs(w_est_prev[k,:])+1e-2)) @ np.sign(np.expand_dims(w_est_prev[l,:],-1))
        w_est_prev[k,:] = w_est[k,:]
        w_est[k,:] = np.squeeze(phi + mu* uki.transpose() @ (D[k,i] - uki @ phi) - mu*lamda*reg_term)


