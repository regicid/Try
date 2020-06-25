import subprocess
import os
PARAM = np.linspace(0,1,100)
for i in range(len(PARAM)//12+1)
	KWARGS = PARAM[12*i:12*(i+1)]
	os.putenv("KWARGS",KWARGS)
	os.putenv("i",i)
	process = subprocess.Popen("srun -N=1 -n=12 -o=~/Results/$i python try.py $KWARGS")
	process.communicate()
