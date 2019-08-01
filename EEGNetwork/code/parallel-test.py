#Runs multiple copies of "Approach-2.py" in parallel
import sys
import subprocess

procs = []
#Specify range based on number of copies that need to be run
#range is also based on which subject the code is being ran on
for i in range(10):
    proc = subprocess.Popen([sys.executable, 'Approach-2.py', str(i)])
    procs.append(proc)

for proc in procs:
    proc.wait()