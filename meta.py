import subprocess
import os
import numpy as np
import pickle
import subprocess
from shlex import split
import math

N = 12
os.system("mkdir Results")
PARAM = np.linspace(0,10,100)
P = np.linspace(.51,.7,10)

for p in P:
	for i in range(math.ceil(len(PARAM)/N)):
		KWARGS = PARAM[N*i:N*(i+1)]
		pickle.dump(KWARGS,open("./KWARGS_"+str(i),"wb"))
		bash = "srun -N 1 --partition=firstgen python try.py "+str(i) + " " + str(p)
		subprocess.Popen(bash.split(),stdout=subprocess.PIPE)	

