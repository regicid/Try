from classs import CurtyMarsili

def compressor(self):
    self.f_history = self.f_history[::100]
    self.α_history = self.α_history[::100]
    self.prop_i = self.prop_i[::100]
    self.q_history = self.q_history[::100]
    self.anti_history = self.anti_history[::100]
    self.N_f= self.N_f[::100]
    self.fitness_history = self.fitness_history[::100,]

import os
import pickle
a = os.listdir("./Results")
for i in a:
    try:
        r = pickle.load(open(i,"rb"))
        r = compressor(r)
        pickle.dump(r,open(f"./Results_clean/{i}","wb"))
