import multiprocessing as mp

value = mp.Value("d", 0)
array = mp.Array("i", [1, 2, 3])
