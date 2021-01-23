
import tensorflow as tf

if __name__ == "__main__":
    chars = '0123456789+ '
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print(indices_char)