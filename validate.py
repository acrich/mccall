import os, sys
import importlib


def run_all():
    tests = os.listdir('validation')

    for test in tests:
        print(test)
        if test == '__pycache__':
            continue
        module = importlib.import_module('validation.' + test.split('.')[0])
        module.main()


if __name__ == '__main__':
    run_all()
