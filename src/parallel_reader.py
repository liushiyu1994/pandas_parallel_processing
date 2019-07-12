import time
import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import multiprocessing.dummy as mt

target_col_list = [1, 5, 10, 20]
thread_num = os.cpu_count()
if thread_num > 10:
    n_row = 30000
    n_col = 30000
    chunksize = 5000
else:
    n_row = 5000
    n_col = 5000
    chunksize = 500
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


def single_pandas_with_appending(df, f_out):
    new_df = df.iloc[:, target_col_list]
    new_df.to_csv(f_out, header=None, index=False)


def check_result(rate):
    with open(raw_csv_file) as f1_in, open(processed_csv_file) as f2_in:
        for line1, line2 in zip(f1_in, f2_in):
            if np.random.random() < rate:
                l1 = line1.split(',')
                l2 = line2.split(',')
                new_col_index = np.random.randint(4)
                raw_col_index = target_col_list[new_col_index]
                num1 = float(l1[raw_col_index])
                num2 = float(l2[new_col_index])
                if abs(num1 - num2) > 1e-10:
                    raise ValueError("Not equal!\nNum1: {}\nNum2: {}".format(num1, num2))
    print("Test passed!")


@clock
def single_processor():
    print("Start processing by Pandas...")
    df_iter = pd.read_csv(raw_csv_file, header=None, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        for input_df in df_iter:
            new_df = single_pandas_func(input_df)
            new_df.to_csv(f_out, header=None, index=False)
    print("Finish processing by Pandas.")


@clock
def thread_parallel_processor():
    print("Start processing by thread-based parallel...")
    df_iter = pd.read_csv(raw_csv_file, header=None, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        with mt.Pool(thread_num) as pool:
            result_iter = pool.imap(single_pandas_func, df_iter)
            for result_df in result_iter:
                result_df.to_csv(f_out, header=None, index=False)
    print("Finish processing by thread-based parallel.")


@clock
def process_parallel_processor():
    print("Start processing by process-based parallel...")
    df_iter = pd.read_csv(raw_csv_file, header=None, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        with mp.Pool(thread_num) as pool:
            result_iter = pool.imap(single_pandas_func, df_iter)
            for result_df in result_iter:
                result_df.to_csv(f_out, header=None, index=False)
    print("Finish processing by process-based parallel.")


@clock
def thread_parallel_with_writer_processor():
    print("Start processing by thread-based parallel with writer...")
    df_iter = pd.read_csv(raw_csv_file, header=None, chunksize=chunksize)
    with open(processed_csv_file, 'w') as f_out:
        pass
    with open(processed_csv_file, 'a') as f_out:
        with mt.Pool(thread_num) as pool:
            pool.map_async(lambda x: single_pandas_with_appending(x, f_out), df_iter)
    print("Finish processing by thread-based parallel with writer.")


def main():
    print("Random number generating")
    large_csv_generator()
    print("Random number generated!")
    single_processor()
    check_result(0.01)
    thread_parallel_processor()
    check_result(0.01)
    process_parallel_processor()
    check_result(0.01)
    thread_parallel_with_writer_processor()
    check_result(0.1)


if __name__ == '__main__':
    main()
