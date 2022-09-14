# Planarian Activity Response Knowledgebase (PARK)
================================

Lightweight repository to explore and statistically compare UV responses. Released in conjuction with the publication "Adaptive robustness through incoherent signaling mechanisms in a regenerating brain". To get started, simply download or clone the repository to your machine. Then access the [compiled data files](https://tinyurl.com/robustBehavior) and down load the objects to the [_data folder_](/data/) in your local repository. Note that this public version of the database contains only published datasets and genetic conditions. For questions on collaboration and/or access to the full database with additional screening results, please contact the repository owner or wangbo@stanford.edu.


### Basic usage
Details of plotting functions are provided in the notebooks in the main directory and vary based on experiment type. A standard usage is shown below. The user provides a list of gene knockdown names to test against the control for a experimental condition. The program the searches the database for appropriate replications of the desired conditions, compiles them, and runs statistical comparisons to the appropriate control.  

![picture](/tools/demo.png)

In the left panel, time series of the response are plotted with statistically significant deviations from the control indicated by the top colored bar. The other panels plot bootstrap distributions of various response measurements described in [here](/tools/measurements_pop.py) and [here](/tools/measurements.py)


## Requirements
Built for python3. All requirements available through conda installation.

```console
numpy
matplotlib
tqdm
```

## Citation


## Authors

* **Samuel Bray**
* **Livia Wyss**
* **Bo Wang**

## License

This project is licensed under GNUV3
