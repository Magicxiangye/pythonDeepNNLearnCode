
import tensorflow as tf
import numpy as np
import re

if __name__ == "__main__":
    sent = 'Where is Sandra?'
    a =[x.strip() for x in re.split('\W+', sent)]
    print(a)
