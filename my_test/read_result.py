# -*- coding: utf-8 -*-
# /usr/bin/python3.6
import re

fr = open("vocab_de.txt", 'r')
vocab_de = eval(fr.read())
# print(vocab_de)
# for de in vocab_de:
    # print(vocab_de[de])
fr = open("vocab_en.txt", 'r')
vocab_en = eval(fr.read())
# print(vocab_en[9])

fr = open("test_01/result.txt", 'r')
result = fr.read()
result = result.split('\n')

input_list = []
target_list = []
predict_list = []
type = 1
# fw = open("test_01/result_text.txt", 'w+')
for line in result:
    line_seq = re.sub("\D", ",", line)
    num = 0
    num_list = []
    is_num = 0

    for ch in line_seq:
        if ch != ',':
            num = num * 10 + int(ch)
            is_num = 1
        elif is_num == 1:
            num_list.append(num)
            is_num = 0
            num = 0
        else:
            num = 0
    # print(num_list)

    if line.find("input") >= 0:
        type = 1
    elif line.find("target") >= 0:
        type = 2
    elif line.find("predict") >= 0:
        type = 3

    if type == 1:
        words = [vocab_de[idx] for idx in num_list]
        input_list.append(words)
        # print("input: %s \n" % words)
        # fw.write("input: %s \n" % words)
    elif type == 2:
        words = [vocab_en[idx] for idx in num_list]
        target_list.append(words)
        # print("target: %s \n" % words)
        # fw.write("target: %s \n" % words)
    elif type == 3:
        words = [vocab_en[idx] for idx in num_list]
        predict_list.append(words)
        # fw.write("predict: %s \n" % words)

fw = open("test_01/result_text.txt", 'w+')
i = 0
# print(input_list)
# print(target_list)
for sentence in input_list:
    fw.write("input: ")
    for word in sentence:
        fw.write("%s " % word)
    fw.write("\n")
    fw.write("target: ")
    for word in target_list[i]:
        fw.write("%s " % word)
    fw.write("\n")
    fw.write("predict: ")
    for word in predict_list[i]:
        fw.write("%s " % word)
    fw.write("\n\n")
    i = i + 1
    # print(line)
# print(result)

