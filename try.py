import numpy as np
import pandas as pd
import pickle
import sys
import multiprocessing as mtp
import os
os.system("mkdir ./Results")

class CurtyMarsili(object):
    def __init__(self,z=0,z2 = 0,z3=0,a = 1, N=5000, p=0.52, m=11,γ = 0.05,γ2 = .05,σ_mut = 3*10**-8,α_dandy = 1,n = 100,Ω = 1,c = .03,selection_force=2,raoult=True):
        #set the parameters
        self.γ2 = γ2
        self.raoult = raoult
        self.N_f = int(np.round(N*z))
        self.N_α = int(np.round(N*z2))
        self.N = N
        self.follower = np.zeros(self.N,dtype="bool")
        self.follower[:self.N_f] = True
        self.α = np.zeros(self.N,dtype='bool')
        self.α[:self.N_α] = True
        self.anti_conformist = np.zeros(self.N,dtype="bool")
        b = np.random.random(size=self.N)
        self.anti_conformist[b<z3] = True
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
        self.D = np.empty(N)
        self.D[self.follower] = np.random.choice([-1,1],p = [0.5, 0.5],size = self.follower.sum())
        self.D[~self.follower] = np.random.choice([-1,1],p = [1-p,p],size = N - self.follower.sum())
        self.q = []
        self.prop_i = []
        self.prop_lazy = []
        self.α_dandy = α_dandy
        in_deg = np.zeros(self.N,dtype="int")
        a = np.unique(self.network,return_counts=True)
        in_deg[a[0]] = a[1]
        self.N_f = [self.N_f]
        self.dandy_share = []
        self.α_history = []
        self.f_history = []
        self.anti_history = []
        self.fitness_history = np.zeros(shape=(0,4))
        self.accuracy = .5*np.ones(shape=(self.N))
        self.q_history = []
    def compute_q(self):
        return np.mean(self.D[self.follower] > 0)
    def compute_pi(self):
        return np.mean(self.D > 0)
    def compute_informed(self):
        return np.mean(self.D[~self.follower] > 0)
    def iterate(self,T=20000): # Iterative imitation process
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
        self.fitness_history = np.vstack((self.fitness_history,np.zeros(shape=(T,4))))
        for t in range(T):
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
            b = np.random.random(size=self.N)
            self.α[b<self.σ_mut] = 1 - self.α[b<self.σ_mut]
            b = np.random.random(size=self.N)
            self.follower[b<self.σ_mut] = 1 - self.follower[b<self.σ_mut]
            if self.raoult:
                b = np.random.random(size=self.N)
                self.anti_conformist[b<self.σ_mut] = 1 - self.anti_conformist[b<self.σ_mut]
            self.accuracy += self.γ2*(self.D>0) -self.γ2*self.accuracy
            self.fitness = self.accuracy + self.Ω*in_deg/self.N - self.c*(~self.follower)
            self.fitness /= self.fitness.sum()
            for j in range(self.selection_force):
                self.selection()
            self.N_f.append(self.follower.sum())
            self.α_history.append(self.α.mean())
            self.f_history.append(self.follower.mean())
            self.anti_history.append(self.anti_conformist.mean())
            self.fitness_history[t,] = [self.fitness[~self.follower*~self.anti_conformist].mean(),self.fitness[~self.α*self.follower*~self.anti_conformist].mean(),self.fitness[self.α*self.follower*~self.anti_conformist].mean(),self.fitness[self.follower*self.anti_conformist].mean()]

            self.iterate()
            self.q_history.append(self.compute_q())
            self.prop_i.append(1-np.mean(self.follower[self.network]))
            self.dandy_share.append(self.follower[self.network[self.follower*self.α,]].mean())
    def selection(self):
        t = len(self.q_history)
        i = np.random.choice(range(self.N))
        j = np.random.choice(range(self.N),p = self.fitness)
        self.α[i] = self.α[j]
        self.network[i,] = self.network[j,]
        self.network_scores[i,] = self.network_scores[j,]
        self.follower[i] = self.follower[j]
        self.anti_conformist[i] = self.anti_conformist[j]
        self.accuracy[i] = self.accuracy[j]

i = sys.argv[1]
p = float(sys.argv[2])
z = pickle.load(open("KWARGS_"+i,"rb"))

def get_cm(o):
    CM = CurtyMarsili(z=.04,z2=.02,z3=.02,Ω = o,γ = .05,c = .01,p = p)
    CM.dynamics(10**6)
    o = np.round(o,2)    
    pickle.dump(CM,open(f"./Results/result_{o}_{p}","wb"))

l = mtp.Pool()
runs = l.map_async(get_cm,z)
l.close()
l.join()
Results = []
for run in runs.get():
    Results.append(run)
