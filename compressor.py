from classs import CurtyMarsili
import pandas as pd
def compressor(self):
    self.f_history = pd.DataFrame(self.f_history).rolling(100).mean().values[::100,0]
    self.α_history = pd.DataFrame(self.α_history).rolling(100).mean().values[::100,0]
    self.prop_i = pd.DataFrame(self.prop_i).rolling(100).mean().values[::100,0]
    self.q_history = self.q_history[::100]
    self.anti_history = pd.DataFrame(self.anti_history).rolling(100).mean().values[::100,0]
    self.N_f= self.N_f[::100]
    self.fitness_history = pd.DataFrame(self.fitness_history).rolling(100).mean().values[::100,:]
    del self.dandy_share
    return r

import os
import pickle
a = os.listdir("./Results")
for i in a:
    try:
        r = pickle.load(open("Results/"+i,"rb"))
        r = compressor(r)
        pickle.dump(r,open(f"./Results_clean/{i}","wb"))
        os.system("rm Results/"+i)
    except:
        print(i)
        os.system("rm Results/"+i)
