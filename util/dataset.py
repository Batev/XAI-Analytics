import enum
import zipfile
import pandas as pd
import logging as log

from statsmodels.api import datasets
from io import StringIO
from urllib.request import urlopen
from shutil import copyfileobj


class Datasets(enum.Enum):
    anes96 = 1
    cancer = 2
    ccard = 3
    china_smoking = 4
    co2 = 5
    committee = 6
    copper = 7
    cpunish = 8
    elnino = 9
    engel = 10
    fair = 11
    fertility = 12
    grunfeld = 13
    heart = 14
    longley = 15
    macrodata = 16
    modechoice = 17
    nile = 18
    randhie = 19
    scotland = 20
    spector = 21
    stackloss = 22
    star98 = 23
    statecrime = 24
    strikes = 25
    sunspots = 26
    census = 27
    cervical_cancer = 28
    yt_spam_comments = 29
    other = 50


class Dataset:

    def __init__(self, id: Datasets, name: str, url: str, df: pd.DataFrame):
        self._id = id
        self._name = name
        self._url = url
        self._df = df

    @classmethod
    def built_in(cls, id: str):

        dataset_id = Datasets[id]
        name = cls.get_name(dataset_id)
        url = cls.get_url(dataset_id)
        df = cls.get_dataframe(dataset_id)

        return cls(dataset_id, name, url, df)

    @classmethod
    def from_url(cls, name: str, url: str):
        dataset_id = Datasets.other
        status, content = is_url_valid(url=url)
        if status:
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        else:
            msg = "Invalid URL to a dataset: {}.".format(url)
            log.error(msg)
            raise ConnectionError(msg)

        return cls(dataset_id, name, url, df)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, new_value):
        self._url = new_value

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_value):
        self._df = new_value

    @staticmethod
    def get_name(id: Datasets) -> str:
        """
        Get the full name for a dataset.
        :param id: The id of the dataset
        :return: The full name
        """

        name = None
        if id.value == 1:
            name = "American National Election Survey 1996"
        elif id.value == 2:
            name = "Breast Cancer Data"
        elif id.value == 3:
            name = "Bill Greene’s credit scoring data."
        elif id.value == 4:
            name = "Smoking and lung cancer in eight cities in China."
        elif id.value == 5:
            name = "Mauna Loa Weekly Atmospheric CO2 Data"
        elif id.value == 6:
            name = "First 100 days of the US House of Representatives 1995"
        elif id.value == 7:
            name = "World Copper Market 1951-1975 Dataset"
        elif id.value == 8:
            name = "US Capital Punishment dataset."
        elif id.value == 9:
            name = "El Nino - Sea Surface Temperatures"
        elif id.value == 10:
            name = "Engel (1857) food expenditure data"
        elif id.value == 11:
            name = "Affairs dataset"
        elif id.value == 12:
            name = "World Bank Fertility Data"
        elif id.value == 13:
            name = "Grunfeld (1950) Investment Data"
        elif id.value == 14:
            name = "Transplant Survival Data"
        elif id.value == 15:
            name = "Longley dataset"
        elif id.value == 16:
            name = "United States Macroeconomic data"
        elif id.value == 17:
            name = "Travel Mode Choice"
        elif id.value == 18:
            name = "Nile River flows at Ashwan 1871-1970"
        elif id.value == 19:
            name = "RAND Health Insurance Experiment Data"
        elif id.value == 20:
            name = "Taxation Powers Vote for the Scottish Parliamant 1997"
        elif id.value == 21:
            name = "Spector and Mazzeo (1980) - Program Effectiveness Data"
        elif id.value == 22:
            name = "Stack loss data"
        elif id.value == 23:
            name = "Star98 Educational Dataset"
        elif id.value == 24:
            name = "Statewide Crime Data 2009"
        elif id.value == 25:
            name = "U.S. Strike Duration Data"
        elif id.value == 26:
            name = "Yearly sunspots data 1700-2008"
        elif id.value == 27:
            name = "Adult census dataset"
        elif id.value == 28:
            name = "Risk Factors for Cervical Cancer"
        elif id.value == 29:
            name = "YouTube Spam Comments"
        else:
            msg = "Invalid dataset id '{}'. Please select a valid dataset.".format(id)
            log.error(msg)
            raise AttributeError(msg)

        return name

    @staticmethod
    def get_url(id: Datasets) -> str:
        """
        Get the website url of the dataset.
        :param id: The id of the dataset
        :return: The url of the dataset
        """
        if id.value <= 26:
            return "http://www.statsmodels.org/dev/datasets/generated/{}.html".format(id.name)
        elif id.value == 27:
            return "https://ethicalml.github.io/xai/index.html?highlight=load_census#xai.data.load_census"
        elif id.value == 28:
            return "https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29#"
        elif id.value == 29:
            return "https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection#"
        else:
            msg = "Invalid dataset id '{}'. Please select a valid dataset.".format(id)
            log.error(msg)
            raise AttributeError(msg)

    @staticmethod
    def get_dataframe(id: Datasets) -> pd.DataFrame:
        if id.value <= 26:
            cmd = "datasets.{}.load_pandas().data".format(id.name)
            return eval(cmd)
        elif id.value == 27:
            from xai.data import load_census
            return load_census()
        elif id.value == 28:
            return Dataset.from_url(
                "Risk Factors for Cervical Cancer",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv").df
        elif id.value == 29:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
            status, content = is_url_valid(url=url)
            if status:
                file_name = url.rsplit('/', 1)[-1]
                with urlopen(url) as in_stream, open(file_name, 'wb') as out_file:
                    copyfileobj(in_stream, out_file)
                zip_file_object = zipfile.ZipFile(file_name, 'r')
                dataframes = []
                unsupported_os = '__MACOSX'
                for file_name in zip_file_object.namelist():
                    if file_name.endswith('.csv'):
                        if not file_name.startswith(unsupported_os):
                            file = zip_file_object.open(file_name)
                            content = file.read()
                            dataframes.append(pd.read_csv(StringIO(content.decode('utf-8'))))
                        else:
                            msg = "{} encoding is not supported, only UTF-8. File '{}' will be ignored"\
                                .format(unsupported_os, file_name)
                            log.warning(msg)
                return pd.concat(dataframes, ignore_index=True)
            else:
                msg = "Invalid URL to a dataset: {}.".format(url)
                log.error(msg)
                raise ConnectionError(msg)
        else:
            msg = "Invalid dataset id '{}'. Please select a valid dataset.".format(id)
            log.error(msg)
            raise AttributeError(msg)


def is_url_valid(url: str, disable_ssl_certificate_validation=True) -> (bool, bytes):
    """
    Check whether an URL is valid.
    If the status code of the HTTP response is less than 400.
    1xx - informational
    2xx - success
    3xx - redirection
    4xx - client error
    5xx - server error
    :param url: The url to be checked.
    :param disable_ssl_certificate_validation: If disable_ssl_certificate_validation is true, SSL cert validation will
     not be performed.
    :return: A tuple (status, content): (True, when the url returns a valid response code, The content of the message)
    """
    import httplib2
    h = httplib2.Http(disable_ssl_certificate_validation=disable_ssl_certificate_validation)
    resp, content = h.request(url)
    status = int(resp['status'])
    return (200 >= status < 300), content
