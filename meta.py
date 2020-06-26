import subprocess
import os
import numpy as np
import pickle
import subprocess
from shlex import split
os.system("mkdir ./Results")
PARAM = np.linspace(0,1,100)
for i in range(len(PARAM)//12+1):
	print(i)
	KWARGS = PARAM[12*i:12*(i+1)]
	pickle.dump(KWARGS,open("./KWARGS_"+str(i),"wb"))
	bash = "srun -N 1 --output=./Results/" + str(i) +" python try.py "+str(i) + " &"
	subprocess.Popen(bash.split(),stdout=subprocess.PIPE)	
