import pandas as pd
import eli5
import enum
import xai
import logging as log

from xai import data
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
from IPython.core.display import display
from ipywidgets import widgets
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NUMERIC_TYPES = ["int", "float"]
RANDOM_NUMBER = 33
EXAMPLES_SPAN_ELI5 = 20
EXAMPLES_SPAN_LIME = 10
EXAMPLES_DIR_LIME = "lime_results"
TEST_SPLIT_SIZE = 0.3

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.DEBUG)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)


class Models(enum.Enum):
    LOGISTIC_REGRESSION = 1
    DECISION_TREE = 2
    RANDOM_FOREST = 3
    XGB = 4


class SplitTypes(enum.Enum):
    IMBALANCED = 1
    BALANCED = 2


class Model:
    def __init__(self, id: int, name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series):
        self._id = id
        self._name = name
        self._model = model
        self._X = X
        self._y = y
        self._X_test = None
        self._y_test = None
        # Frontend Widgets associated with this model.
        self._remove_features_sm = None                 # sm -> Select Multiple
        self._remove_features_button = None
        self._train_model_button = None
        self._model_type_dd = None
        self._split_type_dd = None
        self._cross_columns_sm = None

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
    def model(self):
        return self._model

    @model.setter
    def model(self, new_value):
        self._model = new_value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new_value):
        self._X = new_value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_value):
        self._y = new_value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, new_value):
        self._X_test = new_value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, new_value):
        self._y_test = new_value

    @property
    def remove_features_sm(self):
        return self._remove_features_sm

    @remove_features_sm.setter
    def remove_features_sm(self, new_value):
        self._remove_features_sm = new_value

    @property
    def remove_features_button(self):
        return self._remove_features_button

    @remove_features_button.setter
    def remove_features_button(self, new_value):
        self._remove_features_button = new_value

    @property
    def model_type_dd(self):
        return self._model_type_dd

    @model_type_dd.setter
    def model_type_dd(self, new_value):
        self._model_type_dd = new_value

    @property
    def split_type_dd(self):
        return self._split_type_dd

    @split_type_dd.setter
    def split_type_dd(self, new_value):
        self._split_type_dd = new_value

    @property
    def cross_columns_sm(self):
        return self._cross_columns_sm

    @cross_columns_sm.setter
    def cross_columns_sm(self, new_value):
        self._cross_columns_sm = new_value

    @property
    def train_model_button(self):
        return self._train_model_button

    @train_model_button.setter
    def train_model_button(self, new_value):
        self._train_model_button = new_value


class Split:
    def __init__(self, type: SplitTypes, value: list):
        self._type = type
        self._value = value

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        return self._value


def explain_single_instance(classifier: Pipeline,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            example: int):

    num_features, cat_features = divide_features(X_test)
    new_ohe_features = get_ohe_cats(classifier, cat_features)

    # Transform the categorical feature's labels to a lime-readable format.
    categorical_names = {}
    for col in cat_features:
        categorical_names[X_test.columns.get_loc(col)] = [new_col.split("__")[1]
                                                          for new_col in new_ohe_features
                                                          if new_col.split("__")[0] == col]

    def custom_predict_proba(X, model):
        """
        Create a custom predict_proba for the model, so that it could be used in lime.
        :param X: Example to be classified.
        :param model: The model - classifier.
        :return: The probability that X will be classified as 1.
        """
        X_str = convert_to_lime_format(X, categorical_names, col_names=X_test.columns, invert=True)
        return model.predict_proba(X_str)


    # log.debug("Categorical names for lime: {}".format(categorical_names))

    explainer = LimeTabularExplainer(convert_to_lime_format(X_test, categorical_names).values,
                                     mode="classification",
                                     feature_names=X_test.columns.tolist(),
                                     categorical_names=categorical_names,
                                     categorical_features=categorical_names.keys(),
                                     discretize_continuous=True,
                                     random_state=RANDOM_NUMBER)

    # log.info("Person {}'s data: \n{}".format(example, X_test.iloc[example]))
    log.info("Person {}'s actual result: {}".format(example, y_test[example]))

    custom_model_predict_proba = partial(custom_predict_proba, model=classifier)
    observation = convert_to_lime_format(X_test.iloc[[example], :], categorical_names).values[0]
    explanation = explainer.explain_instance(observation,
                                             custom_model_predict_proba,
                                             num_features=len(num_features))

    return explanation


def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    """Converts data with categorical values as string into the right format
    for LIME, with categorical values as integers labels.
    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency.
    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    """

    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()

    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }

        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }

        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime


def divide_features(df: pd.DataFrame) -> (list, list):
    """
    Separate the numerical from the non-numerical columns of a pandas.DataFrame.
    :param df: The pandas.DataFrame to be separated.
    :return: Two lists. One containing only the numerical column names and another one only
    the non-numerical column names.
    """
    num = []
    cat = []

    for n, t in df.dtypes.items():
        is_numeric = False
        for nt in NUMERIC_TYPES:
            if str(t).startswith(nt):
                is_numeric = True
                num.append(n)
        if not is_numeric:
            cat.append(n)

    return num, cat


def get_column_transformer(numerical: list, categorical: list) -> ColumnTransformer:

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    return ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical),
                    ('cat', categorical_transformer, categorical)])


def get_pipeline(ct: ColumnTransformer, model: Models) -> Pipeline:

    if model is Models.LOGISTIC_REGRESSION:
        return Pipeline([("preprocessor", ct),
                         ("model",
                         LogisticRegression(class_weight="balanced",
                                            solver="liblinear",
                                            random_state=RANDOM_NUMBER))])
    elif model is Models.DECISION_TREE:
        return Pipeline([("preprocessor", ct),
                         ("model", DecisionTreeClassifier(class_weight="balanced"))])
    elif model is Models.RANDOM_FOREST:
        return Pipeline([("preprocessor", ct),
                         ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])
    elif model is Models.XGB:
        # scale_pos_weight to make it balanced
        return Pipeline([("preprocessor", ct),
                         ("model", XGBClassifier(n_jobs=-1))])
    else:
        raise NotImplementedError


def get_split(split: Split, cat_features: list, df_x: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    if split.type is SplitTypes.BALANCED:
        X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, train_idx, test_idx = \
            xai.balanced_train_test_split(
                df_x, df_y, *split.value,
                min_per_group=600,
                max_per_group=600,
                categorical_cols=cat_features)
        return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced
    elif split.type is SplitTypes.IMBALANCED:
        X_train, X_test, y_train, y_test = train_test_split(df_x,
                                                            df_y,
                                                            stratify=df_y,
                                                            test_size=TEST_SPLIT_SIZE,
                                                            random_state=RANDOM_NUMBER)
        return X_train, X_test, y_train, y_test
    else:
        raise NotImplementedError


def get_ohe_cats(model: Pipeline, cat_features: list) -> list:
    preprocessor = model.named_steps["preprocessor"]
    # Get all categorical columns (including the newly generated)
    ohe_categories = preprocessor.named_transformers_["cat"].named_steps['onehot'].categories_
    new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]

    return new_ohe_features


def get_all_features(model: Pipeline, num_features: list, cat_features: list) -> list:
    return num_features + get_ohe_cats(model, cat_features)


def train_model(model_type: Models, split: Split, df_x: pd.DataFrame, df_y: pd.Series) -> \
        (Pipeline, pd.DataFrame, pd.Series):

    num_features, cat_features = divide_features(df_x)

    log.debug("Numerical features: {}".format(num_features))
    log.debug("Categorical features: {}".format(cat_features))

    # Transform the categorical features to numerical
    preprocessor = get_column_transformer(num_features, cat_features)

    model = get_pipeline(preprocessor, model_type)

    X_train, X_test, y_train, y_test = get_split(split, cat_features, df_x, df_y)

    # Now we can fit the model on the whole training set and calculate accuracy on the test set.
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)
    log.info("Model accuracy: {}".format(accuracy_score(y_test, y_pred)))
    log.info("Classification report: \n{}".format(classification_report(y_test, y_pred)))

    return model, X_test, y_test


def interpret_model(model: Pipeline, num_features, cat_features):
    return eli5.show_weights(model.named_steps["model"],
                             feature_names=get_all_features(model,
                                                            num_features,
                                                            cat_features))

#######################################################################################################################


class OutputWidgetHandler(log.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '100%',
            'height': '160px',
            'border': '1px solid black'
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def get_logs(self):
        """ Return the logs """
        result = ''
        for line in self.out.outputs:
            result = result + line['text']

        return result

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()


def get_dataset_as_dataframe(name: str) -> pd.DataFrame:
    df = data.load_census()
    return df


def get_grid_template_columns(number_of_models: int, min_number: int) -> str:
    grid_template_columns = ''
    if number_of_models > min_number:
        part = 100.0/number_of_models
        for i in range(number_of_models):
            grid_template_columns = grid_template_columns + str(part) + "% "

    else:
        grid_template_columns = '33% 33% 33%'

    return grid_template_columns


def get_model_by_id(models: list, id: int) -> Model:
    model = None
    for m in models:
        if m.id == id:
            model = m

    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_remove_features_button(models: list, button: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.remove_features_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_train_model_button(models: list, button: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.train_model_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_split_type_dd(models: list, dropdown: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.split_type_dd is dropdown:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model
