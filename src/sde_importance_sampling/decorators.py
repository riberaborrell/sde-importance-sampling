import datetime
import time

def timer(function):
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        m, s = divmod(t1 -t0, 60)
        h, m = divmod(m, 60)
        print('Total time running: {:d}:{:02d}:{:02.2f}'.format(int(h), int(m), s))

        return result
    return function_timer
