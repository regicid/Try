from classs import CurtyMarsili
import os
import pickle
import numpy as np
Results = []
a = os.listdir()
for i in a:
	try:
		r = pickle.load(open(i,"rb"))
		Results.append(r)
	except:
		print(i)


R = np.zeros((10,len(Results)))

for i in range(len(Results)):
	a = np.mean(Results[i].q_history[-300000:])
	b = Results[i].anti_conformist.mean()
	c = (Results[i].follower*Results[i].α).mean()
	d = (Results[i].follower*~Results[i].α).mean()
	e = (~Results[i].follower).mean()
	f = Results[i].c
	g = Results[i].Ω
	h = np.mean(Results[i].prop_i[-300000:])
	j = np.abs(np.array(Results[i].q_history[-300000:]) - .5).mean()
	p = Results[i].p
	R[:,i] = np.array([a,b,c,d,e,f,g,h,j,p])

print(list(R))