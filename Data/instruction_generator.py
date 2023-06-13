import os
import os.path as osp
import multiprocessing as mp

manager = mp.Manager
ClassDictionary = manager().dict()

class Method(object):
    def __init__(self):
        self.name = ''
        self.ClassName = ''
        self.instructions = []
    
    def add_instruction(self, raw_strig: str):
        self.instructions.append(raw_strig.strip())


class SmaliClass(object):
    def __init__(self):
        self.name = ''
        self.methods = []
        self.api_names = []
    
    def add_method(self, method: Method):
        self.methods.append(method)

    def add_api_name(self, api_name: str):
        self.api_names.append(api_name)


def FunctionGenerator(SmaliFile):

    MethodFlag = False

    for line in open(SmaliFile, 'r').readlines():
        if line.startswith('.class'):
            ClassName = line.strip().split(' ')[-1][1:-1]
            if ClassName in ClassDictionary:
                ClassDictionary[ClassName] += 1
                break
            else:
                ClassDictionary[ClassName] = 1
        if line.startswith('.method'):
            MethodFlag = True
            method = Method()
            method.name = line.split(' ')[-1][:-1]
            method.ClassName = ClassName
            continue
        if line.startswith('.end method'):
            MethodFlag = False
            yield method
        if MethodFlag and len(line.strip()) > 0 and not line.strip().startswith('.'):
            method.add_instruction(line)


def SmaliInstructionGenerator(SmaliRootDir, flag='method'):
    '''
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    flag: can only be 'method' or 'class' indicating the generator to yield method or class.
    '''

    assert flag in {'method', 'class'}
    
    SmaliFileList = []
    for root, _, files in os.walk(SmaliRootDir, topdown=False):
        SmaliFileList = SmaliFileList + [osp.join(root, x) for x in files if x.endswith('.smali')]
    
    for SmaliFile in SmaliFileList:
        if flag == 'class':
            Class = SmaliClass() 
        for method in FunctionGenerator(SmaliFile):
            if flag == 'method':
                yield method
            else:
                Class.add_method(method)
        if flag == 'class' and len(Class.methods):
            Class.name = Class.methods[0].ClassName
            yield Class
                

if __name__ == "__main__":
    pass