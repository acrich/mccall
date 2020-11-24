import os, sys
import importlib

tests = os.listdir('validation')

for test in tests:
    print(test)
    if test == '__pycache__':
        continue
    module = importlib.import_module('validation.' + test.split('.')[0])
    module.main()
