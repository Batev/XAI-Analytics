"""Historical integrations excluded from the supported Python 3.12 environment."""


def unavailable(feature):
    raise NotImplementedError(
        f"{feature} is unavailable in the modern environment. "
        "See README.md, Legacy integrations, for status and alternatives."
    )


def imbalance_plot(*args, **kwargs):
    unavailable("xai exploratory plots")


def correlations(*args, **kwargs):
    unavailable("xai correlation plots")
