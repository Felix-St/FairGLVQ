# FairGLVQ
Repository for corresponding paper available [here](https://doi.org/10.1007/978-3-031-67159-3_17).

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

# Citation
```
@InProceedings{FairGLVQ,
author="St{\"o}rck, Felix
and Hinder, Fabian
and Brinkrolf, Johannes
and Paassen, Benjamin
and Vaquet, Valerie
and Hammer, Barbara",
title="FairGLVQ: Fairness in Partition-Based Classification",
booktitle="Advances in Self-Organizing Maps, Learning Vector Quantization, Interpretable Machine Learning, and Beyond",
year="2024",
publisher="Springer Nature Switzerland",
pages="141--151",
isbn="978-3-031-67159-3"
}
``` 

# License
MIT license - See [LICENSE](LICENSE).
