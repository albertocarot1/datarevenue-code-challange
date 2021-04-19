# Create dataset task

This script will split the dataset, and process it so that it is ready to be trained/tested.

To see in details why the data is processed this way, check the notebook `DataExploration.ipynb`

```
Usage: dataset.py [OPTIONS]

  Split the wine dataset in train and test, and use the first one to teach a
  data processor how to transform the data in a way that it can be used with
  the chosen model.

  Then transform the two splits, so that they can be used to train, and
  successively validate, the wine rating prediction model.

Options:
  --in-csv TEXT        Path to csv file that contains the wine dataset.
                       [required]

  --test-perc INTEGER  Percentage of the data set to hold out for final
                       validation  [required]

  --out-dir TEXT       Directory where the two processed data splits will be
                       saved.  [required]

  --drop-duplicates    Whether duplicates in the datset should be dropped or
                       not.

  --help               Show this message and exit.

```
