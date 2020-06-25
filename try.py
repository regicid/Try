import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import multiprocessing

class CurtyMarsili(object):
    def __init__(self,z=0,z2 = 0,a = 1, N=2000, p=0.52, m=11,γ = 0.1,σ_mut = 10**-8,α_dandy = 1,n = 100,Ω = 1,c = .03,selection_force=1):
        #set the parameters
        
        self.N_f = int(np.round(N*z))
        self.N_α = int(np.round(N*z2))
        self.N = N
        self.follower = np.zeros(self.N,dtype="bool")
        self.follower[:self.N_f] = True
        self.α = np.zeros(self.N,dtype='bool')
        self.α[:self.N_α] = True
        self.anti_conformist = np.zeros(self.N,dtype="bool")
        self.Ω = Ω
        self.p = p
        self.m = m
        self.a = a
        self.c = c
        self.n = n
        self.selection_force = selection_force
        self.σ_mut = σ_mut
        self.γ = γ # Memory loss of partner's performance
        self.network = np.empty(shape = (self.N,m),dtype = "int16")
        self.network_scores = np.zeros(shape = (self.N,m))
        for i in range(self.N):
            self.network[i,] = np.random.choice(np.delete(range(N),i),size=m,replace=False)
        #pick the initial choices for informed and non-informed agents
        self.D = np.empty(N)
        self.D[self.follower] = np.random.choice([-1,1],p = [0.5, 0.5],size = self.follower.sum())
        self.D[~self.follower] = np.random.choice([-1,1],p = [1-p,p],size = N - self.follower.sum())
        self.α_history = np.zeros(shape=(0,self.N),dtype='bool')
        self.D_history = np.zeros(shape=(0,self.N),dtype='bool')
        self.f_history = np.zeros(shape=(0,self.N),dtype='bool')
        self.fitness_history = np.zeros(shape=(0,3))
        self.q = []
        self.prop_i = []
        self.prop_lazy = []
        self.α_dandy = α_dandy
        in_deg = np.zeros(self.N,dtype="int")
        a = np.unique(self.network,return_counts=True)
        in_deg[a[0]] = a[1]
        self.in_d = in_deg
        self.N_f = [self.N_f]
        self.dandy_share = []
        
    def compute_q(self):          
        return np.mean(self.D[self.follower] > 0)   
    def compute_pi(self):
        return np.mean(self.D > 0)   
    def compute_informed(self):
        return np.mean(self.D[~self.follower] > 0)        
    def iterate(self,T=10000): # Iterative imitation process
        self.D[self.follower] = np.random.choice([-1,1],p = [0.5, 0.5],size = self.follower.sum())
        self.D[~self.follower] = np.random.choice([-1,1],p = [1-self.p,self.p],size = self.N - self.follower.sum())
        if self.follower.sum()>0:
            for t in range(T//self.n):
                #pick a random follower
                random_follower = np.random.choice(np.where(self.follower)[0],size = self.n)
                #get the choices of the group
                group_choices = self.D[self.network[random_follower,]]
                #align your choice with that of the majority
                avg_group_choice = np.mean(group_choices,axis=1)
                self.D[random_follower] = np.sign(avg_group_choice)*(1-2*self.anti_conformist[random_follower])    
    def dynamics(self,T):
        self.α_temporary = np.zeros(shape=(T,self.N),dtype='bool')
        self.f_temporary = np.zeros(shape=(T,self.N),dtype='bool')
        self.D_temporary = np.zeros(shape=(T+1000,self.N),dtype='bool')
        self.fitness_temporary = np.zeros(shape=(T,3))
        if len(self.q)>1000:
            self.D_temporary[:1000,] = self.D_history[-1000:,]
        self.q_temporary = []
        for t in tqdm(range(T)):
            # Now we update the networks (record scores, and get rid of the worst network member)
            in_deg = np.zeros(self.N,dtype="int")
            a = np.unique(self.network,return_counts=True)
            in_deg[a[0]] = a[1]
            #self.in_d = np.vstack((self.in_d,in_deg))
            self.network_scores += (self.D[self.network]>0) + self.α_dandy*np.broadcast_to(self.α,shape=(self.m,self.N)).transpose()*(self.D[self.network]
                                    - np.mean(self.D[self.network]))**2 - self.γ*self.network_scores
            a = np.where(np.random.random(size=self.N)< self.γ)[0]
            weakest_link = np.argmin(self.network_scores[a,],axis=1)
            p = in_deg + self.a
            for i in range(len(a)):
                I = a[i]
                not_listened = np.where(~np.isin(np.arange(self.N),np.append(self.network[I,],I)))[0]
                p2 = p[not_listened]
                p2 = p2/p2.sum()
                self.network[I,weakest_link[i]] = np.random.choice(not_listened,p = p2)
                self.network_scores[I,weakest_link[i]] = self.network_scores[I,].mean()
            if len(self.q) > 1000 or len(self.q_temporary) > 1000:
                b = np.random.random(size=self.N)
                self.α[b<self.σ_mut] = 1 - self.α[b<self.σ_mut]
                b = np.random.random(size=self.N)
                self.follower[b<self.σ_mut] = 1 - self.follower[b<self.σ_mut]
                self.fitness = self.D_temporary[t:t+1000,].mean(axis=0) + self.Ω*in_deg/self.N - self.c*(~self.follower)
                #p -= p.min()/2
                self.fitness /= self.fitness.sum()
                for j in range(self.selection_force):
                    self.selection()
                self.N_f.append(self.follower.sum())
                self.α_temporary[t,]= self.α
                self.f_temporary[t,]= self.follower
                self.fitness_temporary[t,] = [self.fitness[~self.follower].mean(),self.fitness[~self.α*self.follower].mean(),self.fitness[self.α*self.follower].mean()]
                self.prop_lazy.append(np.mean((self.follower*~self.α)[self.network])/(self.follower*~self.α).sum())
            self.iterate()
            self.q_temporary.append(self.compute_q())
            self.prop_i.append(1-np.mean(self.follower[self.network]))
            self.dandy_share.append(self.follower[self.network[self.follower*self.α,]].mean())
            self.D_temporary[1000+t,]= self.D>0
    def selection(self):
        t = len(self.q_temporary)
        i = np.random.choice(range(self.N))
        j = np.random.choice(range(self.N),p = self.fitness)
        self.α[i] = self.α[j]
        self.network[i,] = self.network[j,]
        self.network_scores[i,] = self.network_scores[j,]
        self.follower[i] = self.follower[j]
        self.D_temporary[t:t+1000,i] = self.D_temporary[t:t+1000,j]
    def record(self):
        t = len(self.q_temporary)
        F = self.D_temporary[t:t+1000,]
        self.α_temporary = self.α_temporary[::10,]
        self.f_temporary = self.f_temporary[::10,]
        self.fitness_temporary = self.fitness_temporary[::10,]
        self.q_temporary = self.q_temporary[::10]
        a = len(self.q_temporary)
        self.α_history = np.vstack((self.α_history,self.α_temporary[:a,]))
        self.f_history = np.vstack((self.f_history,self.f_temporary[:a,]))
        self.fitness_history = np.vstack((self.fitness_history,self.fitness_temporary[:a,]))
        self.D_history = F
        self.q = self.q + self.q_temporary`

i = sys.argv[1]
z = pickle.load(open("KWARGS_"+i,"rb"))

def get_cm(o):
    CM = CurtyMarsili(z=.04,z2=.02,Ω = 0.7,γ = o)
    CM.dynamics(10**6)
    CM.record()
    return CM

l = mtp.Pool()
runs = l.map_async(get_cm,z)
l.close()
l.join()
Results = []
for run in runs.get():
    Results.append(run)
