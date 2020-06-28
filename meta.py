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
for i in range(math.ceil(len(PARAM)/N)):
	print(i)
	KWARGS = PARAM[N*i:N*(i+1)]
	pickle.dump(KWARGS,open("./KWARGS_"+str(i),"wb"))
	bash = "srun -N 1 --partition=secondgen --output=./Results/" + str(i) +" python try.py "+str(i) + " &"
	subprocess.Popen(bash.split(),stdout=subprocess.PIPE)	
