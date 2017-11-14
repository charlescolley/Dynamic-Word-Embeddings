import multiprocessing as mp
import psutil as p

def process_func():
  name = mp.current_process().name
  print "hello world from worker {} with memory {}".format(name,
                                                           p.virtual_memory().free/1e6)
