from itertools import zip_longest
from subprocess import Popen, STDOUT

limit = 40


remain_commands = [
'python admission_swap_batch_lime.py 0.8 21 22'
]


commands3 = ['python admission_swap_batch_lime.py 0.6 ' + str(i) + ' ' + str(i+1) for i in range(50, 100)]
commands2 = ['python admission_swap_batch_lime.py 0.8 ' + str(i) + ' ' + str(i+1) for i in range(50, 100)]
commands1 = ['python admission_swap_batch_lime.py 0.8 ' + str(i) + ' ' + str(i+1) for i in range(0, 50)]

commands = commands1 + commands2 + commands3 + remain_commands
print(commands)

groups = [(Popen(command, stderr=STDOUT)
          for command in commands)] * limit # itertools' grouper recipe
for processes in zip_longest(*groups): # run len(processes) == limit at a time
    for p in filter(None, processes):
        p.wait()