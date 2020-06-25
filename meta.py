os.putenv("KWARGS",PARAM[i:i+12])


subprocess.Popen(srun -N=1 -n=12 python try.py $KWARGS)