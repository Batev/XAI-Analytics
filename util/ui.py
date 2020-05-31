import logging as log

from ipywidgets import widgets, Layout, ButtonStyle, Label, GridBox, Button, HBox
from util.commons import get_model_by_id, FeatureImportanceType, PDPType
from util.split import SplitTypes


def generate_analyze_grid(
        show_imbalance_selectmultiple,
        show_imbalance_button,
        correlations_dendogram_button,
        correlations_matrix_button) -> GridBox:

    return GridBox(children=[
        Label(layout=Layout(width='auto', height='auto'), value='Features to analyze: '),
        Button(layout=Layout(width='auto', height='auto'), disabled=True, style=ButtonStyle(button_color='white')),
        Button(layout=Layout(width='auto', height='auto'), disabled=True, style=ButtonStyle(button_color='white')),
        show_imbalance_selectmultiple,
        Button(layout=Layout(width='auto', height='auto'), disabled=True, style=ButtonStyle(button_color='white')),
        Button(layout=Layout(width='auto', height='auto'), disabled=True, style=ButtonStyle(button_color='white')),
        show_imbalance_button,
        correlations_dendogram_button,
        correlations_matrix_button
        ],
        layout=Layout(
            width='auto',
            grid_template_columns='33% 33% 33%',
            align_items='center',
            # grid_template_columns='auto auto auto',
            # grid_template_rows='auto auto auto',
            grid_gap='2px 1px'))


def add_dummy_widgets(min_number, children, number_of_models):
    for i in range(min_number - number_of_models):
        children.append(widgets.Button(layout=Layout(width='auto', height='auto'), disabled=True,
                                       style=ButtonStyle(button_color='white')))


def get_grid_template_columns(number_of_models: int, min_number: int) -> str:
    grid_template_columns = ''
    if number_of_models > min_number:
        part = 100.0/number_of_models
        for i in range(number_of_models):
            grid_template_columns = grid_template_columns + str(part) + "% "

    else:
        grid_template_columns = '33% 33% 33%'

    return grid_template_columns


def generate_model_grid(df_X,
                        number_of_models,
                        models,
                        on_click_feature_exclude_button,
                        on_value_change_split_type_dropdown,
                        on_click_model_train_button):

    df_X_columns = list(df_X.columns)
    len_df_X_columns = len(df_X_columns)
    children = []
    min_number = 3

    # Row 1
    for i in range(number_of_models):
        children.append(
            Label(layout=Layout(width='auto', height='auto'), value='Remove features for model {}'.format(i + 1)))
    # Row 1: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 2
    for i in range(number_of_models):
        w = widgets.SelectMultiple(options=df_X_columns, rows=len_df_X_columns if len_df_X_columns <= 20 else 20,
                                   layout=Layout(width='auto', height='auto'))
        model = get_model_by_id(models, i)
        model.remove_features_sm = w
        children.append(w)
    # Row 2: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 3
    for i in range(number_of_models):
        w = widgets.Button(description='Remove features', disabled=False, button_style='danger', tooltip='Click me',
                           icon='trash', layout=Layout(width='auto', height='auto'))
        w.on_click(on_click_feature_exclude_button)
        model = get_model_by_id(models, i)
        model.remove_features_button = w
        children.append(w)
    # Row 3: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 4:
    for i in range(number_of_models):
        children.append(Label(layout=Layout(width='auto', height='auto'), value='Train model {}'.format(i + 1)))
    # Row 4: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 5:
    for i in range(number_of_models):
        model = get_model_by_id(models, i)
        w = widgets.Dropdown(options=model.model_type.algorithm_options, description='Model type:', disabled=False,
                             layout=Layout(width='auto', height='auto'))
        model.model_type_dd = w
        children.append(w)
    # Row 5: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 6:
    for i in range(number_of_models):
        w = widgets.Dropdown(options=[s.name for s in SplitTypes], description='Train/Test split type:', disabled=False,
                             layout=Layout(width='auto', height='auto'),
                             description_tooltip='Splits the features and the target into train/test split training '
                                                 'sets with a balanced number of examples for each of the categories of'
                                                 ' the columns provided. For example, if the columns provided are '
                                                 '“gender” and “loan”, the resulting splits would contain an equal '
                                                 'number of examples for Male with Loan Approved, Male with '
                                                 'Loan Rejected, Female with Loan Approved, and Female with '
                                                 'Loan Rejected.')
        model = get_model_by_id(models, i)
        w.observe(on_value_change_split_type_dropdown, names='value')
        model.split_type_dd = w
        children.append(w)
    # Row 6: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 7:
    for i in range(number_of_models):
        children.append(
            Label(layout=Layout(width='auto', height='auto'), value='Cross columns for model {}'.format(i + 1)))
    # Row 7: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 8:
    for i in range(number_of_models):
        model = get_model_by_id(models, i)
        w = widgets.SelectMultiple(options=model.X, rows=8 if len_df_X_columns <= 20 else 20,
                                   layout=Layout(width='auto', height='auto'), description='', disabled=True,
                                   description_tooltip='One or more positional arguments (passed as *args) '
                                                       'that are used to split the data into the cross product '
                                                       'of their values.')
        model.cross_columns_sm = w
        children.append(w)
    # Row 8: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    # Row 9:
    for i in range(number_of_models):
        w = widgets.Button(description='Train model', disabled=False, button_style='success', tooltip='Click me',
                           icon='cogs', layout=Layout(width='auto', height='auto'))
        w.on_click(on_click_model_train_button)
        model = get_model_by_id(models, i)
        model.train_model_button = w
        children.append(w)
    # Row 9: add dummy widgets
    add_dummy_widgets(min_number, children, number_of_models)

    return GridBox(children=children,
                   layout=Layout(
                           width='auto',
                           grid_template_columns=get_grid_template_columns(number_of_models, min_number),
                           align_items='center',
                           # grid_template_columns='auto auto auto',
                           grid_template_rows='auto auto auto',
                           grid_gap='1px 1px'))


def init_strip_eq_radio(on_value_change_eq_radio) -> widgets:
    eq_radio = widgets.RadioButtons(options=['>', '=', '<'], value='=')
    eq_radio.observe(on_value_change_eq_radio, names='value')

    return eq_radio


def init_strip_value_slider(on_value_change_value_slider, min_val: float, max_val: float, step: float) -> widgets:
    value_slider = widgets.IntSlider(value=min_val, min=min_val, max=max_val, step=step, disabled=False,
                                     continuous_update=False, orientation='horizontal', readout=True,
                                     readout_format='d', layout=Layout(width='214px', height='auto'))
    value_slider.observe(on_value_change_value_slider, names='value')

    return value_slider


def init_strip_value_select_dropdown(on_value_change_value_select_dropdown, options: list) -> widgets:
    value_select_dropdown = widgets.Dropdown(options=options, value=None,
                                             layout=Layout(width='214px', height='auto'))
    value_select_dropdown.observe(on_value_change_value_select_dropdown, names='value')

    return value_select_dropdown


def generate_reset_strip_hbox(on_click_reset_button):

    children = []

    reset_button = widgets.Button(disabled=False, style=ButtonStyle(button_color='gray'),
                                  tooltip='Reset the dataset to its initial state',
                                  icon='undo', layout=Layout(width='auto', height='auto'))
    reset_button.on_click(on_click_reset_button)
    children.append(reset_button)
    stripped_columns_label = Label(layout=Layout(width='auto', height='auto'),
                                   value='Stripped columns for the dataset: ')
    children.append(stripped_columns_label)

    return HBox(children=children,
                layout=Layout(
                           width='auto',
                           grid_template_columns="50% 50%",
                           align_items='center',
                           # grid_template_columns='auto auto auto',
                           grid_template_rows='auto',
                           grid_gap='3px 3px'))


def generate_feature_importance_grid(models: list) -> GridBox:
    children = []

    # Row 1
    children.append(
        Label(layout=Layout(width='auto', height='auto'), value='Choose a feature importance method'))
    children.append(
        Label(layout=Layout(width='auto', height='auto'), value='Choose one or more model(s)'))

    # Row 2
    # if you change the description of this widget,
    # you have to also adjust it in the notebook function call.
    children.append(
        widgets.Select(
            description='Type',
            options=[elem.name for elem in FeatureImportanceType],
            disabled=False)
    )
    # if you change the description of this widget,
    # you have to also adjust it in the notebook function call.
    children.append(
        widgets.SelectMultiple(
            description='Model(s)',
            options=[model.name for model in models],
            disabled=False)
    )

    return GridBox(children=children,
                   layout=Layout(
                           width='auto',
                           grid_template_columns="50% 50%",
                           grid_template_rows='auto',
                           align_items='center',
                           grid_gap='3px 3px'))


def generate_pdp_grid(models: list) -> GridBox:
        children = []

        # Row 1
        children.append(
            Label(layout=Layout(width='auto', height='auto'), value='Choose a PDP method'))
        children.append(
            Label(layout=Layout(width='auto', height='auto'), value='Choose one or more model(s)'))

        # Row 2
        # if you change the description of this widget,
        # you have to also adjust it in the notebook function call.
        children.append(
            widgets.Select(
                description='Type',
                options=[elem.name for elem in PDPType],
                disabled=False)
        )
        # if you change the description of this widget,
        # you have to also adjust it in the notebook function call.
        children.append(
            widgets.SelectMultiple(
                description='Model(s)',
                options=[model.name for model in models],
                disabled=False)
        )

        return GridBox(children=children,
                       layout=Layout(
                           width='auto',
                           grid_template_columns="50% 50%",
                           grid_template_rows='auto',
                           align_items='center',
                           grid_gap='3px 3px'))


def generate_pdp_feature_selection_grid(models: list) -> GridBox:
    children = []

    # Row 1
    children.append(
        Label(layout=Layout(width='auto', height='auto'), value='Feature 1 for ...'))
    children.append(
        Label(layout=Layout(width='auto', height='auto'), value='(optional) Feature 2 for ...'))

    for model in models:
        # Row 2 -> Row (2 + len(models))
        # if you change the description of this widget,
        # you have to also adjust it in the notebook function call.
        features = model.features_ohe.copy()
        features.insert(0, 'None')

        children.append(
            widgets.Select(
                description="... " + model.name,
                options=model.features_ohe,
                disabled=False))
        children.append(
            widgets.Select(
                description="... " + model.name,
                options=features,
                value='None',
                disabled=False))

    return GridBox(children=children,
                   layout=Layout(
                       width='auto',
                       grid_template_columns="50% 50%",
                       grid_template_rows='auto',
                       align_items='center',
                       grid_gap='3px 3px'))


def get_reset_strip_hbox_label(hbox: HBox) -> Label:

    for child in hbox.children:
        if isinstance(child, Label):
            return child

    return None
