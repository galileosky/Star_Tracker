import time
import subprocess as sp
import multiprocessing


def square(x):
    time.sleep(1/x)
    print(f'x =  {x}')

    return x * x

numbers = [1, 2, 3, 4, 5]

#squared_numbers_map = map(square, numbers)



n_cores = multiprocessing.cpu_count()
n_cores = 1

print(f'n_cores =  {n_cores}\n')

# call call_match_list() to let Match do the work
#
if n_cores == 1:
    squared_numbers_map = map(square, numbers)
else:
    pool = multiprocessing.Pool(n_cores)

    print(f'pool =  {pool}\n')

    squared_numbers_map = pool.map(square, numbers)





print(f'squared_numbers_map =  {squared_numbers_map}')

# Convert the map object to a list to see the results
squared_numbers_list = list(squared_numbers_map)
print(squared_numbers_list)





print()
