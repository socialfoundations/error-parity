# Example jupyter notebooks

| File | Metric | Dataset | Description |
| -------- | -------- | -------- | -- |
| [relaxed-equalized-odds.usage-example-folktables.ipynb](relaxed-equalized-odds.usage-example-folktables.ipynb) | equalized odds | ACSIncome | Example usage of `RelaxedThresholdOptimizer` to map Pareto frontier of attainable fairness-accuracy trade-offs for a given predictor. |
| [parse-folktables-datasets.ipynb](parse-folktables-datasets.ipynb) | - | ACSIncome / folktables | Notebook that downloads and parses folktables datasets (required to run the folktables/ACSIncome examples). |
| [relaxed-equalized-odds.usage-example-synthetic-data.ipynb](relaxed-equalized-odds.usage-example-synthetic-data.ipynb) | equalized odds | synthetic (no downloads necessary) | Stand-alone example on synthetic data. |
| [usage-example-for-other-constraints.synthetic-data.ipynb](usage-example-for-other-constraints.synthetic-data.ipynb) | TPR equality, FPR equality, demographic parity | synthetic (no downloads) | Stand-alone example with other available fairness metrics (based on TPR, FPR, or PPR). |
| [example-with-postprocessing-and-inprocessing.ipynb](example-with-postprocessing-and-inprocessing.ipynb) | equalized odds | synthetic (no downloads) | Example of using relaxed postprocessing with an in-processing fairness algorithm. |
| [brute-force-example_equalized-odds-thresholding.ipynb](brute-force-example_equalized-odds-thresholding.ipynb) | equalized odds | synthetic (no downloads) | Comparison between using the `RelaxedThresholdOptimizer` and a brute-force solver (out of curiosity). |
