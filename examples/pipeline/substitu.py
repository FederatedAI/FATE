import os
import sys
import random


def get_filelist(path):
    files = []

    for home, dirs, file in os.walk(path):
        for filename in file:
            files.append(os.path.join(home, filename))

    return files


def substi(file):
    import json
    fin = open(file, "r")
    fout = open(file + ".bak", "w")
    for line in fin:
        if line.find("component_param(") != -1:
            line = line.replace("component_param(", "component_param(")
        fout.write(line)

    os.system("mv " + file + ".bak " + file)

if __name__ == "__main__":
    files = get_filelist(sys.argv[1])

    for file in files:
        substi(file)

