import os, sys
import importlib

tests = os.listdir('validation')

for test in tests:
    print(test)
    module = importlib.import_module('validation.' + test.split('.')[0])
    module.main()
