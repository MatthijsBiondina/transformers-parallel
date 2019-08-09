import datetime
import os
from tqdm import tqdm
import traceback
import shutil
import sys


class Tools:

    @staticmethod
    def pyout(*args, end='\n'):
        tqdm.write(datetime.datetime.now().strftime(
            "%d-%m-%Y %H:%M"), end=' ')
        for arg in args:  # list(map(str, args)):
            if not isinstance(arg, str):
                arg = '\033[94m' + str(arg) + '\033[0m'
            for ii, string in enumerate(arg.split('\n')):
                if ii == len(arg.split('\n')) - 1:
                    tqdm.write(string, end=' ')
                else:
                    tqdm.write(string)
        tqdm.write('', end=end)

    @staticmethod
    def trace(*args, end='\n', ex=None):
        Tools.pyout("DEBUG ->", traceback.format_stack()[-2].split('\n')[0])
        Tools.pyout("------->  ", *args, end=end)
        if ex is not None:
            sys.exit(ex)

    @staticmethod
    def makedirs(path, delete=False):
        try:
            os.makedirs(path)
        except FileExistsError:
            if delete:
                shutil.rmtree(path)
                os.makedirs(path)

    @staticmethod
    def poem(iterable, description="PROGRESS", total=None):
        pbar = tqdm(iterable, total=total)
        if len(description) <= 23:
            pbar.set_description(description + ' ' * (23 - len(description)))
        else:
            pbar.set_description(description[:20] + '...')
        return pbar
