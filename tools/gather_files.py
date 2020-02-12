import os
import shutil
import argparse

def gather_files(oldDir, newDir):
    if oldDir is None:
        raise ValueError('Old Directory is None!')
    if newDir is None:
        raise ValueError('New Directory is None!')
    
    for root, dirs, files in os.walk(oldDir):
        for file in files:
            if (1 - file.endswith('.jpg')) and (1 - file.endswith('.png')):
                continue
            
            oldfile = os.path.join(root, file)
            newfile = os.path.join(newDir, file)
            shutil.copyfile(oldfile, newfile)
            print('copy {} to {}'.format(oldfile, newfile))

    print('gather files finished!')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("oldDir", type=str, help="input old path")
    parse.add_argument("newDir", type=str, help="input new path")
    args = parse.parse_args()
    gather_files(args.oldDir, args.newDir)