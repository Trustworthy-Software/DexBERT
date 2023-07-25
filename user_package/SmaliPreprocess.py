import os
import os.path as osp

from instruction_generator import SmaliClass, Method

def FunctionGenerator(SmaliFile, OnlyFunc=True):
    
    MethodFlag = False

    for line in open(SmaliFile, 'r').readlines():
        if line.startswith('.class'):
            ClassName = line.strip().split(' ')[-1][1:-1]
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
            if OnlyFunc:
                if line.strip().startswith('invoke'):
                    line = line.split(' ')[-1].strip()
                    if line not in method.instructions:
                        method.add_instruction(line)
            else:
                line = line.strip()
                if line not in method.instructions:
                    method.add_instruction(line)
                

def SmaliClassInstructionGenerator(SmaliRootDir, OnlyFunc=True):
    '''
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    '''
    
    SmaliFileList = []
    for root, _, files in os.walk(SmaliRootDir, topdown=False):
        SmaliFileList = SmaliFileList + [osp.join(root, x) for x in files if x.endswith('.smali')]
    
    for SmaliFile in SmaliFileList:
        Class = SmaliClass() 
        for method in FunctionGenerator(SmaliFile, OnlyFunc):
            Class.add_method(method)
        if len(Class.methods):
            Class.name = Class.methods[0].ClassName
            for instruction in method.instructions:
                if instruction not in Class.api_names:
                    Class.add_api_name(instruction)
            yield Class

def Smalis2Txt(TxtRootDir, SmaliDir, only_keep_func_name=False):
    '''
    This method extracts Smali Classes in different smali files and save them in a txt file.
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    TxtRootDir: is the root directory to save generated txt files.
    '''
    ApkName = SmaliDir.split('/')[-1] if SmaliDir.split('/')[-1] else SmaliDir.split('/')[-2]
    with open(osp.join(TxtRootDir, ApkName+'.txt'), 'w') as f:
        for cls in SmaliClassInstructionGenerator(SmaliDir, OnlyFunc=only_keep_func_name):
            f.write('ClassName: ' + cls.name + '\n')
            if only_keep_func_name:
                for api_name in cls.api_names:
                    f.write(api_name+'\n')
            else:
                for method in cls.methods:
                    f.write('MethodName: ' + method.name + '\n')
                    for instruction in method.instructions:
                        f.write(instruction+'\n')
                    f.write('\n')
            f.write('\n')
            f.write('ClassEnd\n\n')

if __name__ == '__main__':

    pass
    
