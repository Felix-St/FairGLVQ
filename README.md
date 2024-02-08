# FairGLVQ
Preliminary repository for corresponding paper.


# Structure
The source code is contained in the [src](src) folder.
The implementation of FairGLVQ resides in [src/models/GLVQ_models.py](src/models/GLVQ_models.py).

To install the required packages:
``` 
pip install -r requirements.txt
```

To rerun the synthetic experiments: 
``` 
python benchmark_synthetics.py --dataset XOR
python benchmark_synthetics.py --dataset Local
```

To rerun the real-world experiments: 
``` 
python benchmark_models.py --dataset COMPAS
python benchmark_models.py --dataset ADULT
```

Note that the real-world experiments might take some time to complete.
The results used in the paper can be found in [results](results)
where the results of real-world benchmarks are saved after completion.
Modify [src/analyse_benchmark.py](src/analyse_benchmark.py) to visualize these results.

# License
MIT license - See [LICENSE](LICENSE).