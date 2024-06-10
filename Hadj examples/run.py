import subprocess
import sys

#this small code shows how you can run a pytho programme from a file 
#we ru the programme and redirect its output to a system Pipe.
#We then read from the pipe to get the output and display it

p = subprocess.Popen([sys.executable, "Hadj examples/myprogram.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(line)
retval = p.wait()
