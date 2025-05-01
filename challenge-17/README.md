# Sorting on a Systolic Array

## Overview

Standard bubble sort has a Big O runtime of O(n^2). Using a 1-dimensional systolic array with parallel pairwise comparisons, we can expect a worst case runtime of O(n). As noted by ChatGPT, the number of comparisons is still O(n^2), but the reduction in execution time or cycles comes from the parallelization.

## Results

Future work
- It would be interesting to go back and fix the threaded Python version and benchmark that instead of the basic sequential Python bubble sort
