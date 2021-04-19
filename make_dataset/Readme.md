# Create dataset task

This script will split the dataset, and process it so that it is ready to be trained/tested.


```
Usage: dataset.py [OPTIONS]

  Open a wine dataset csv, split it into train and test set, and process it
  so that it can be fed into a model for training/test.

  Parameters
  ----------
  
  in_csv: path to the csv file on local disk. 
  test_perc: integer defining the percentage of dataset that must be hold out for test.
  out_dir: directory where train/test csv files should be saved to.
  drop_duplicates: whether duplicates should be dropped from the dataset or not
  
  Returns
  ------- 
  None

Options:
  --in-csv 
  --test-perc
  --out-dir
  --drop-duplicates
  --help          Show this message and exit.
```
