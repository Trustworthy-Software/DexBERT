import sys
import os
import os.path as osp
from time import time

def Disassemble(ApkPath, OutDir):
    '''
    To disassemble Dex bytecode in a given Apk file into smali code.
    Java version: "11.0.11" 2021-04-20
    baksmali tool was downloaded on: https://bitbucket.org/JesusFreke/smali/downloads/
    '''
    os.system("java -jar {} disassemble {} -o {}".format(osp.join(sys.path[0], 'baksmali-2.5.2.jar'), ApkPath, OutDir)) 

if __name__ == "__main__":
    ApkPath = sys.argv[1]
    OutDir  = sys.argv[2]
    time_start = time()
    Disassemble(ApkPath=ApkPath, OutDir=OutDir)
    print("Time cost: ", time()-time_start)
