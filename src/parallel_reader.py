import time
import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import multiprocessing.dummy as mt

target_col_list = [1, 5, 10, 20]
thread_num = os.cpu_count()
if thread_num > 10:
    n_row = 20000
    n_col = 20000
else:
    n_row = 5000
    n_col = 5000
chunksize = int(n_row / thread_num) / 2
raw_csv_file = './data/large_csv.csv'
processed_csv_file = './data/processed_csv.csv'


def large_csv_generator():
    random_num = np.random.random([n_row, n_col])
    np.savetxt(raw_csv_file, random_num, delimiter=',')


def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s)' % (elapsed, name, arg_str))
        return result
    return clocked


def single_pandas_func(df):
    return df.iloc[:, target_col_list]


@clock
def single_processor():
    input_df = pd.read_csv(raw_csv_file)
    new_df = single_pandas_func(input_df)
    new_df.to_csv(processed_csv_file, index=False)


@clock
def thread_parallel_processor():
    df_iter = pd.read_csv(raw_csv_file, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        with mt.Pool(thread_num) as pool:
            result_iter = pool.imap(single_pandas_func, df_iter)
            for result_df in result_iter:
                result_df.to_csv(f_out, header=False)


@clock
def process_parallel_processor():
    df_iter = pd.read_csv(raw_csv_file, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        with mp.Pool(thread_num) as pool:
            result_iter = pool.imap(single_pandas_func, df_iter)
            for result_df in result_iter:
                result_df.to_csv(f_out, header=False)


def main():
    large_csv_generator()
    single_processor()
    thread_parallel_processor()
    process_parallel_processor()


if __name__ == '__main__':
    main()
