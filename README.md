# ADA helpers
Helper functions for applied data analysis.

## Structure

* `helpers/` contains all the scripts. To use them:
  1. Copy this folder to the root of your project
  2. In your code, you can them import them using `from helpers.<...> import *`, where `<...>` should be the name of one of the `.py` files of the `helpers` folder. For example:
     ```py
     from helpers.dataframes import *
     from helpers.texts import *
     ```
* `test_datasets/` simply contains some datasets that are used in the tests and demonstrations of this repository.
* `test_<...>.ipynb` files are Jupyter notebooks that demonstrate the use of the functions of this repository.
