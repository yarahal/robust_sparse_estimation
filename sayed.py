import numpy as np
from math import sqrt


class Network:
    def set_random_network(self,N):
        # define network
        self.N = N;
        self.A = np.random.randint(0,5,size=(N,N)) # A(l,k): weight of edge directed from l to k
        A = A / np.sum(A,axis=0)
    
    def set_random_model(self,M):
        self.M = M
        self.w = np.random.randint(low=0,high=10,size=(M,1))
        return self.w
    
    def make_model_sparse(self):
        mask = np.random.randint(low=0,high=2,size=(self.M,1))
        self.w = self.w * mask
        return self.w

    def collect_data(self,var_v,Nt):
        #  collect data from every agent
        #  every agent probes the model
        self.var_v = var_v
        self.Nt = Nt
        self.U = np.zeros(shape=(self.N,Nt)) # for recording agent inputs
        self.D = np.zeros(shape=(self.N,Nt)) # for recording agent outputs
        for k in range(0,self.N):
            uk = np.random.randn(1,Nt) # generate input according to standard gaussian
            dk = np.zeros(shape=(1,Nt)) 
            vk = sqrt(var_v)*np.random.randn(1,Nt)
            for i in range(self.M,Nt):
                uki = np.flip(uk[:,i-self.M+1:i+1],axis=1)
                dk[:,i] = uki @ self.w + vk[:,i] # convolve and add white gaussian noise
            self.U[k,:] = uk
            self.D[k,:] = dk
    
    def atc(self,mu):
        # diffusion-adaptation
        self.mu = mu
        w_est_prev = np.random.randn(self.N,self.M)
        phi = np.copy(w_est_prev)
        w_est = np.zeros(shape=(self.N,self.M)) # hold estimates

        for i in range(self.M,self.Nt):
            for k in range(0,self.N):
                w_est[k,:] = np.zeros(shape=(1,self.M))
                uki = np.expand_dims(np.flip(self.U[k,i-self.M+1:i+1]),0)  
                grad = -np.squeeze(uki.transpose() @ (self.D[k,i] - uki @ np.expand_dims(w_est_prev[k,:],-1)))      
                phi[k,:] = w_est_prev[k,:] - mu* grad
                for l in range(0,self.N):
                    w_est[k,:] += self.A[l,k] * phi[l,:]
                w_est_prev[k,:] = w_est[k,:]
        
        return w_est
    
    def atc_sparsity(self,mu,lamda):
        # diffusion-adaptation with sparsity
        self.mu = mu
        self.lamda = lamda # regularization parameter
        w_est_prev = np.random.randn(self.N,self.M)
        phi = np.copy(w_est_prev)
        w_est = np.zeros(shape=(self.N,self.M)) # hold estimates

        for i in range(self.M,self.Nt):
            for k in range(0,self.N):
                w_est[k,:] = np.zeros(shape=(1,self.M))
                uki = np.expand_dims(np.flip(self.U[k,i-self.M+1:i+1]),0)  
                reg_term = np.diag(1/(np.abs(w_est_prev[k,:])+1e-2)) @ np.sign(np.expand_dims(w_est_prev[l,:],-1))
                grad = -np.squeeze(uki.transpose() @ (self.D[k,i] - uki @ np.expand_dims(w_est_prev[k,:],-1)))
                phi[k,:] = w_est_prev[k,:] - mu*grad   - mu*lamda*reg_term    
                for l in range(0,self.N):
                    w_est[k,:] += self.A[l,k] * phi[l,:]
                w_est_prev[k,:] = w_est[k,:]
        
        return w_est
    
    def atc_robust(self,mu):
        # diffusion-adaptation with l1 norm
        self.mu = mu
        w_est_prev = np.random.randn(self.N,self.M)
        phi = np.copy(w_est_prev)
        w_est = np.zeros(shape=(self.N,self.M)) # hold estimates

        for i in range(self.M,self.Nt):
            for k in range(0,self.N):
                w_est[k,:] = np.zeros(shape=(1,self.M))
                uki = np.expand_dims(np.flip(self.U[k,i-self.M+1:i+1]),0)  
                grad = np.squeeze(np.sign(self.D(k,i)-uki @ np.expand_dims(w_est_prev[k,:],-1)) * (-uki))
                phi[k,:] = w_est_prev[k,:] - mu * grad 
                for l in range(0,self.N):
                    w_est[k,:] += self.A[l,k] * phi[l,:]
                w_est_prev[k,:] = w_est[k,:]
        
        return w_est
    
