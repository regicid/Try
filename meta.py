import subprocess
import os
import numpy as np
import pickle
import subprocess
from shlex import split
import math

N = 16
os.system("mkdir ./Results")
PARAM = np.linspace(0,1,80)
C = [0,.01,.03,.05]

for c in C:
	for i in range(math.ceil(len(PARAM)/40)):
		print(c)
		KWARGS = PARAM[40*i:40*(i+1)]
		pickle.dump(KWARGS,open("./KWARGS_"+str(i),"wb"))
		bash = "srun -N 1 --partition=dellgen --output=./Results/" + str(i) +" python try.py "+str(i) + " " + str(c)
		subprocess.Popen(bash.split(),stdout=subprocess.PIPE)	

