
import tensorflow as tf
import numpy as np
import re

if __name__ == "__main__":
    sent = ['Where is Sandra?', 'Where','qqq']
    vo = set()
    for a in sent:
        vo = set(a)

    print(vo)
