# -*- coding: utf-8 -*-
# /usr/bin/python3.6
import re
import matplotlib.pyplot as plt

import numpy as np

fr = open("test_02/log.txt", 'r')
lines = fr.read().split('\n')
steps = []
losses = []
acc_steps = []
accuracies = []
for line in lines:
    print(line)
    if line.find("loss") >= 0:
        line = line.split(':')
        steps.append(line[1])
        losses.append(float(line[3]))
    elif line.find("Epoch") >= 0:
        line = line.split(' ')
        step = line[4].split(':')
        step = step[1].split('/')
        acc_steps.append(step[0])
        accuracy = line[6].split('=')
        accuracies.append(float(accuracy[1]))
# print(steps)
# print(losses)
# print(acc_steps)
# print(accuracies)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(steps, losses)
ax2 = fig.add_subplot(2,1,2)
ax2.plot(acc_steps, accuracies)

plt.show()