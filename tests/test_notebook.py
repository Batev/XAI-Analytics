from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell


def test_offline_notebook():
    notebook = nbformat.read(Path(__file__).parent / "notebooks/smoke.ipynb", as_version=4)
    shell = InteractiveShell()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            result = shell.run_cell(cell.source)
            result.raise_error()


def test_main_notebook_controls(data):
    from types import SimpleNamespace

    X, y = data
    notebook = nbformat.read(Path(__file__).parent.parent / "XAI-Analytics.ipynb", as_version=4)
    shell = InteractiveShell()
    # Run the actual setup and control cells with local data in place of a download.
    for index in (1, 3):
        # The test uses Agg; inline renderer setup belongs to the Jupyter kernel.
        shell.run_cell(
            notebook.cells[index].source.replace("%matplotlib inline", "pass")
        ).raise_error()
    shell.user_ns.update(dataset=SimpleNamespace(df=X.assign(target=y)), df_X=X, df_y=y)
    for index in (5, 7, 9, 11, 13, 15, 17):
        # The test uses Agg; inline renderer setup belongs to the Jupyter kernel.
        shell.run_cell(
            notebook.cells[index].source.replace("%matplotlib inline", "pass")
        ).raise_error()
    assert all(
        "census" not in option.lower()
        for option in shell.user_ns["dataset_select_dropdown"].options
    )
    assert shell.user_ns["show_imbalance_button"].disabled
    model = shell.user_ns["models"][0]
    shell.user_ns["change_cross_columns_status"](model, "BALANCED")
    assert model.cross_columns_sm.disabled

    from functools import partial

    from util.commons import fill_model

    shell.user_ns["fill_model"] = partial(
        fill_model, search_options={"param_grid": {}, "cv": 2, "n_jobs": 1}
    )
    model.model_type_dd.value = "DECISION_TREE"
    shell.user_ns["on_click_model_train_button"](model.train_model_button)
    assert model.model is not None
    assert len(model.predictions) == len(model.X_test)
