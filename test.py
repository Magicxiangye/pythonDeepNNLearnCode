
import tensorflow as tf

if __name__ == "__main__":
    chars = '0123456789+ '
    test = set()
    for char in chars:
        test |= set(char)
        print(test)
