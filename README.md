# Reader Benchmark for Large CSV Files


Pandas is a common toolkit for CSV readers. However, its performance is usually not satisfactory for large files. This project tests several different ways to find the best processing protocol for large CSV files.


## Requirements

`numpy` and `pandas` packages are required. The script is tested on `python 3.6` with `numpy=1.16.2` and `pandas=0.23.3`.

## Get started

Just simply run: 

```
python src/parallel_reader.py
```

First, a test data set, CSV file containing random float numbers is generated. Then, several methods are utilized to extract four columns in the data set, and output to a new CSV file. Running time of different methods will be measured and displayed. After running, a sampling test is executed to check if output strictly keep the order of input. The methods include:

1. Ordinary Pandas I/O method.
2. Pandas I/O method with multi-thread acceleration. Writing part is in the main thread.
3. Pandas I/O method with multi-process acceleration. Writing part is in the main process.
4. Pandas I/O method with multi-process acceleration. Writing part is in the individual process.
5. Normal for-loop method.
6. For-loop method with multi-thread acceleration. Writing part is in the main thread.
7. For-loop method with multi-process acceleration. Writing part is in the main process.

The output should be something similar with:

```
Random number generating          
Random number generated!
Start processing by Pandas in single processor...
Finish processing by Pandas in single processor.
[510.94331532s] Pandas in single processor
Test passed!
Start processing by Pandas with thread-based parallel...
Finish processing by Pandas with thread-based parallel.
[502.02295056s] Pandas with thread-based parallel
Test passed!
Start processing by Pandas with process-based parallel...
Finish processing by Pandas with process-based parallel.
[520.49486474s] Pandas with process-based parallel
Test passed!
Start processing by Pandas with thread-based parallel writer...
Finish processing by Pandas with thread-based parallel writer.
[505.13706774s] Pandas with thread-based parallel writer
src/parallel_reader.py:75: UserWarning:
Pandas with thread-based parallel writer:
Not equal! Num1: 0.5488233632170664 Num2: 0.4035580894763974
  warnings.warn("\n{}:\nNot equal! Num1: {} Num2: {}".format(function_name, num1, num2))
Start processing by for-loop in single processor...
Finish processing by for-loop in single processor.
[74.52978328s] for-loop in single processor
Test passed!
Start processing by for-loop with thread-based parallel...
Finish processing by for-loop with thread-based parallel.
[63.97761573s] for-loop with thread-based parallel
Test passed!
Start processing by for-loop with process-based parallel...
Finish processing by for-loop with process-based parallel.
[68.88769884s] for-loop with process-based parallel
Test passed!
```

The warning shows that Pandas with thread-based parallel writer cannot keep order of input files.

## Conclusions

+ For-loop based method is usually much faster than those with Pandas, could be up to 10 times in very large CSV file.
+ In methods using for-loop, thread-based parallel is slightly better than process-based parallel, which are both significantly better than single process in very large CSV file.
+ Process-based for-loop parallel may be better than thread-based parallel in data processing with more complicated computation. 
+ Writing function cannot be parallel, or the order of final files will be different.


## Authors

+ **Shiyu Liu** - *Initial work* - [liushiyu1994](https://github.com/liushiyu1994)


## License

This software is released under the [MIT License](LICENSE-MIT).