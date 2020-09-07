from classs import CurtyMarsili
import os
import pickle
import numpy as np
Results = []
path = "Results"
a = os.listdir(path)
for i in a:
	try:
		r = pickle.load(open(path+"/"+i,"rb"))
		Results.append(r)
	except:
		print(i)


R = np.zeros((10,len(Results)))

for i in range(len(Results)):
	a = np.mean(Results[i].q_history[-10000:])
	b = Results[i].anti_conformist.mean()
	c = (Results[i].follower*Results[i].α).mean()
	d = (Results[i].follower*~Results[i].α).mean()
	e = (~Results[i].follower).mean()
	f = Results[i].c
	g = Results[i].Ω
	h = np.mean(Results[i].prop_i[-10000:])
	j = np.abs(np.array(Results[i].q_history[-10000:]) - .5).mean()
	p = Results[i].p
	R[:,i] = np.array([a,b,c,d,e,f,g,h,j,p])

print(list(R))
np.save("result2.npy",R)
