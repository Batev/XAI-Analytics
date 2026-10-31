def plot_single_pdp_with_pdpbox(model: Model, feature: str, plot_lines=True,
                                x_quantile=True, show_percentile=True, plot_pts_dist=True,)\
        -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for a single feature for a given model.
    :param model: The model for which a PDP should be created
    :param feature: Feature or feature list to investigate, for one-hot encoding features, feature list is required
    :param plot_lines: Whether to plot out the individual lines
    :param x_quantile: Whether to construct x axis ticks using quantiles
    :param show_percentile: Whether to display the percentile buckets, for numeric feature when grid_type='percentile'
    :param plot_pts_dist: Whether to show data points distribution
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """

    # model.model[1] to get the actual model from the pipeline
    pdp_isolate_out = pdp.pdp_isolate(
        model=model.model[1],
        dataset=model.X_test_ohe,
        model_features=model.features_ohe,
        feature=feature)

    fig, axes = pdp.pdp_plot(
        pdp_isolate_out=pdp_isolate_out,
        feature_name=feature if isinstance(feature, str) else feature[0].split('_')[0].title,
        plot_lines=plot_lines,
        x_quantile=x_quantile,
        show_percentile=show_percentile,
        plot_pts_dist=plot_pts_dist,
        frac_to_plot=0.5)

    if feature in model.numerical_features:
        ticks_pdp_ax_vals = axes['pdp_ax']['_pdp_ax'].xaxis.major.formatter.seq
        inverse_ticks = inverse_scale(model, feature, ticks_pdp_ax_vals, 2)
        ticks_pdp_ax = axes['pdp_ax']['_pdp_ax'].get_xticks().tolist()
        for i in range(len(ticks_pdp_ax)):
            ticks_pdp_ax[i] = inverse_ticks[i]
        axes['pdp_ax']['_pdp_ax'].set_xticklabels(ticks_pdp_ax)

        ticks_count_ax_vals = axes['pdp_ax']['_count_ax'].xaxis.major.formatter.seq
        ticks_count_ax = axes['pdp_ax']['_pdp_ax'].get_xticks().tolist()
        for i in range(len(ticks_count_ax)-1):
            ticks_count_ax[i] = \
                ticks_count_ax_vals[i]\
                .replace(
                    "{}, {}".format(str(ticks_pdp_ax_vals[i]), str(ticks_pdp_ax_vals[i+1])),
                    "{}, {}".format(str(inverse_ticks[i]), str(inverse_ticks[i+1])))
        axes['pdp_ax']['_count_ax'].set_xticklabels(ticks_count_ax)

    return fig, axes


def inverse_scale(model: Model, feature: str, values: list, rounding: int) -> list:
    dt = {}
    arr_inv = np.unique(
        model
        .model[0]
        .named_transformers_["num"]
        .named_steps['scale']
        .inverse_transform(
            model.X_test_ohe[model.numerical_features])[:, model.numerical_features.index(feature)])
    arr = np.unique(model.X_test_ohe[feature].array)
    for i in range(len(arr)):
        num = arr[i]
        if rounding:
            num = round(arr[i], rounding)
        dt[num] = arr_inv[i]
    res = []
    for value in values:
        res.append(dt[float(value)])

    return res


def plot_multi_pdp_with_pdpbox(model: Model, feature1: str, feature2: str, plot_type='contour',
                               x_quantile=False, plot_pdp=False) -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for two features for a given model.
    :param model: The model for which a PDP should be created
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted
    :param plot_type: Type of the interact plot, can be 'contour' or 'grid'
    :param x_quantile: Whether to construct x axis ticks using quantiles
    :param plot_pdp: Whether to plot pdp for each feature
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """
    features_to_plot = [feature1, feature2]

    df_temp1 = model.X_test_ohe.copy()
    inverse_values1 = inverse_scale(model, feature1, list(df_temp1[feature1]), 0)
    inverse_values2 = inverse_scale(model, feature2, list(df_temp1[feature2]), 0)
    df_temp1[feature1] = np.array(inverse_values1)
    df_temp1[feature2] = np.array(inverse_values2)
    # model.model[1] to get the actual model from the pipeline
    pdp_interact_out = pdp.pdp_interact(
        model=model.model[1],
        # dataset=model.X_test_ohe,
        dataset=model.X_test,
        model_features=model.X_test.columns,
        features=features_to_plot,
        data_transformer=model
        .model[0]
        .transform)

    fig, axes = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=features_to_plot,
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp)

    return fig, axes