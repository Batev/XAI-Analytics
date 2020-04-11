# XAI-Analytics

XAI-Analytics is a tool for analyzing datasets, interpreting machine learning models and compering their results in form of an IPython Notebook. This tool uses [interactive widgets](https://ipywidgets.readthedocs.io/en/stable/) and is therefore advisable to be run in [binder](http://mybinder.org/). For a quick demo of XAI-Analytics jump to the [demo](#demo) section.

## Getting started

### Run Notebook online on _binder_

1. Start the binder notebook server - [Start Server!](https://mybinder.org/v2/gh/Batev/XAI-Analytics/477b66b97505a0ac92962bcd73ad55e49e2e1a68)
2. Run *XAI-Analytics.ipynb* notebook file

### Run Notebook locally

1. Checkout the XAI-Analytics repository
1. Navigate to checkout directory in _Bash_ or _Powershell_ and ...

    * Unix

        ```bash
        # Start a virtual environment, e.g. venv - path to virtual environment
        $> source /venv/bin/activate
        # Install project requirements
        $> pip3 install -r /path/to/requirements.txt
        # Start notebook server
        $> jupyter notebook
        ```

    * Windows

        ```powershell
        # Start a virtual environment, e.g. venv - path to virtual environment
        $> .\venv\Scripts\activate
        # Install project requirements
        $> pip install -r /path/to/requirements.txt
        # Start notebook server
        $> jupyter notebook
        ```

1. Default browser will be started in current directory
1. Run *XAI-Analytics.ipynb* notebook file

## Prerequisites

XAI-Analytics uses libraries for analyzing datasets like xai and for interpreting machine learning models like eli5, lime, alibi and others. Please refer to the [requirements file](requirements.txt) for a full list of prerequisites.

* [xai](https://github.com/EthicalML/xai)
* [alibi](https://github.com/SeldonIO/alibi)
* [lime](https://github.com/marcotcr/lime)
* [eli5](https://github.com/TeamHG-Memex/eli5/)
* ...

## Goals of XAI-Analytics 1.0

* Analyzing datasets - show feature imbalances, show feature correlations  
* Train models (up to 8 simultaneously on the "same" dataset) with different properties - features, model type, data (e.g. data may be upsampled, or adversarial data may be added)
* Analyze the trained models globally and compare them
* Analyze single examples locally and compare them
* Supports both classification and regression problems

### Non-Goals for version 1.0

* Train models on different datasets simultaneously (e.g. train one model on mnist dataset and other on enron and compare them, it is only possible to train models on the same datasets, where the data could be manipulated)
* Support for semi-supervised, unsupervised and reinforcement learning

## <a name="demo"></a>Demo

_This is an early prototype of the tool._

![early-demo](xai-analytics-demo.gif)

## Known Issues

* Error when training a model on a dataset without any categorical columns (or only one categorical column that is the target)

## TODO

* Add auto-generated adversarial examples to some dataset.
* Add XAI-Analytics to [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks) when ALPHA version is released.
* ...

## License

[BSD 3-Clause License](LICENSE)
