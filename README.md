[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Batev/XAI-Analytics/master)

# XAI-Analytics

XAI-Analytics is a tool that opens the black-box of machine learning. It makes model interpretability easy. 
**XAI-Analytics** offers a wide range of features such as data visualization, data preprocessing, ML-model training and global and local ML-model interpreting.
The interactive Jupyter Notebook offers an user-friendly interface that gives the user full control over the tool. 
For a quick demo of **XAI-Analytics** jump to the [demo](#demo) section. For examples please refer to [examples](#examples).

## Getting started

### Run on _binder_

1. Start the binder notebook server by clicking [here](https://mybinder.org/v2/gh/Batev/XAI-Analytics/master) 
2. Run *XAI-Analytics.ipynb* notebook file

### <a name="run-locally"></a>Run locally

1. Clone the **XAI-Analytics** repository: https://github.com/Batev/XAI-Analytics.git
1. Navigate to checkout directory in _Bash_ or _Powershell_ and ...

    * Unix

        ```bash
        # (Optional) Create a new virtual environment 
        $> python3 -m venv venv
        # Start the virtual environment, e.g. venv
        $> source venv/bin/activate
        # Install project requirements
        $> pip3 install -r requirements.txt
        # Modules with outdated dependencies (e.g. Skater on wordcloud==1.3.1), that are erroneous when installed with pip. 
        $> pip3 install --no-dependencies --no-cache-dir -r requirements-no-deps.txt 
        # Skater only supports scikit-learn<=0.22.2.post1 but other modules require scikit-learn>=0.23.1, therefore this workaround is necessary.
        $> find . -name 'tree_visualizer.py' -exec sed -i 's/from sklearn.externals.six import StringIO/from six import StringIO/g' {} +
        # Start notebook server
        $> jupyter notebook
        ```

    * Windows

        ```powershell
        # (Optional) Create a new virtual environment 
        $> python -m venv venv
        # Start the virtual environment, e.g. venv
        $> .\venv\Scripts\activate
        # Install project requirements
        $> pip install -r requirements.txt
        # Modules with outdated dependencies (e.g. Skater on wordcloud==1.3.1), that are erroneous when installed with pip.  
        $> pip install --no-dependencies --no-cache-dir -r requirements-no-deps.txt
        # Skater only supports scikit-learn<=0.22.2.post1 but other modules require scikit-learn>=0.23.1, therefore this workaround is necessary.
        $> get-content Lib\site-packages\skater-1.1.2-py3.8.egg\skater\core\visualizer\tree_visualizer.py | %{$_ -replace "from sklearn.externals.six","import StringIO/from six import StringIO"}
        # Start notebook server
        $> jupyter notebook
        ```

1. Default browser will be started in current directory
1. Run *XAI-Analytics.ipynb* notebook file

## Prerequisites

Please refer to the [requirements file](requirements.txt) for a list of the prerequisites. The file excludes [XAI](https://github.com/EthicalML/xai) and [Skater](https://github.com/oracle/Skater) because of incompatible versions in the dependencies when started with [binder](https://mybinder.org/). The [requirements.txt](requirements.txt) is auto-generated from [requirements.in](requirements.in) (which consists of the raw dependencies) using [pip-compile](https://github.com/jazzband/pip-tools). For using locally with all required modules please refer to the [Run locally](#run-locally) step.

## <a name="demo"></a>Demo

_v1.0_

[![Demo](https://img.youtube.com/vi/x-FImaqYme8/0.jpg)](https://www.youtube.com/watch?v=x-FImaqYme8)

##  <a name="examples"></a>Examples
Example notebooks can be found in [examples](https://github.com/Batev/XAI-Analytics/tree/master). Quick HTML overview of the example notebooks:
* [Adult Census Income](https://htmlpreview.github.io/?https://github.com/Batev/XAI-Analytics/blob/master/XAI-Example-Income.html)
* [Risk Factors for Cervical Cancer](https://htmlpreview.github.io/?https://github.com/Batev/XAI-Analytics/blob/master/XAI-Example-Cervical.html)

## Known Issues

* ~~Error when training a model on a dataset without any *categorical* columns (or only one categorical column that is the target)~~ -> Fixed by [b87a276](https://github.com/Batev/XAI-Analytics/commit/b87a2769cc3e817be45c39f95321a0bd0873d5b2).
* ~~Internal *PDPBox* error when plotting a PDP: https://github.com/SauceCat/PDPbox/issues/40~~ -> Workaround by [6143001](https://github.com/Batev/XAI-Analytics/commit/61430014c91ee6a69e26930fec6d9b20693fe321).
* *Skater* cannot be installed with *pip*: https://github.com/amueller/word_cloud/issues/134 and https://github.com/oracle/Skater/issues/308
* ~~Not all interpretation techniques work with *XGBoost*.~~ Only *ELI5* does not work with *XGBoost* -> until [this pull request](https://github.com/TeamHG-Memex/eli5/pull/407) is merged.

### TODO

* ~~Add XAI-Analytics to [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks) when ALPHA version is released.~~
* Add auto-generated adversarial examples to some dataset.
* ~~Fix *XGBoost* interpretations bug.~~
* Add option for removing features from a dataset in the preprocessing step.
* Add option for normalizing/removing unknown values from a dataset in the preprocessing step.
* Add custom exceptions and raise them where needed.
* Improve logging -> do not show info messages in the notebooks, use display instead. Add more debug messages.

### Non-Goals

* Support for semi-supervised, unsupervised and reinforcement learning.
* Model interpretation of regression algorithms.

## Creators

This project was created by [Evgeni Batev](https://www.linkedin.com/in/evgeni-batev-49124b40/) under the supervision of [Ao.univ.Prof. Dr. Andreas Rauber](http://www.ifs.tuwien.ac.at/~andi/) at the [Vienna University of Technology](https://www.tuwien.at/en/).

## License

[BSD 3-Clause License](LICENSE)
