import pandas as pd
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import numpy as np
import warnings
import logging
from itertools import product

logger = logging.getLogger(__name__)


def plot_measures(
        results, measures, x_var, split_var,
        kwarg_func=None, kwargs=None, fig=None):
    """ A function for plotting certain pandas DataFrames.

    Assumes the ``results`` data frame has a certain format. In particular,
    each row in ``results`` should correspond to a single trial or simulation.
    Each field/column should do one of two things: describe the context of the
    trial (such as one of the independent variables the trial was run with),
    or record some aspect of the result of the trial.

    ``results`` is expected to have simple integer indices, each
    trial/simulation having its own index.

    ``measures`` should be a list of names of result fields in ``results``.
    For each entry in results, this function will create a separate plot,
    with the plots stacked vertically in the figure.

    ``x_var`` should be a string giving the name of one of the context fields
    in ``results`` which will be used as the x-axis for all plots.

    ``split_var`` should be a string or list of strings, each string giving
    the name of one of the context fields in ``results``. For each unique
    value of ``split_var`` found in ``results`` (or combination of values,
    n the case that a list of strings is supplied), a separate line will be
    created in each of the plots. All trials which have identical values for
    both ``x_var`` and ``split_var`` will be averaged over, and the mean will
    be plotted along with bootstrapped 95% confidence intervals.

    Parameters
    ----------
    results: str or DataFrame
        Data to plot. Each row is a trial. If a string, assumed to be the name
        of a csv file which will be loaded from disk.
    measures: str or list of str
        Create a separate subplot for each entry.
    x_var: str
        Name of the variable to plot along x-axis.
    split_var: str or (list of str)
        Each in the plots corresponds to a unique value of the ``split_var``
        field in results.
    kwarg_func: func (optional)
        A function which accepts one of the values from ``split_var`` and
        returns a dict of key word arguments for the call to plt.errorbar
        for that value of ``split_var``.
    kwargs: dict
        Additional key word args for used for every call to plt.errorbar.
        Arguments obtained by calling ``kwarg_func`` will overwrite the
        args in ``kwargs`` if there is a conflict.
    fig: matplotlib Figure (optional)
        A figure to plot on. If not provided, current figure is used.

    """
    if isinstance(results, str):
        results = pd.read_csv(results)
    if isinstance(measures, str):
        measures = [measures]
    if isinstance(split_var, str):
        split_var = [split_var]

    if fig is None:
        fig = plt.gcf()

    logger.info(results)
    logger.info(results.describe())

    fig.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    n_plots = len(measures)
    for i, measure_name in enumerate(measures):
        ax = fig.add_subplot(n_plots, 1, i+1)
        _plot_measure(
            results, measure_name, x_var, split_var,
            kwarg_func, kwargs, ax=ax)

        if i == 0:
            ax.legend(
                loc='center left', bbox_to_anchor=(1, 0.5),
                prop={'size': 10}, handlelength=3.0, handletextpad=.5,
                shadow=False, frameon=False)

        if i == n_plots - 1:
            ax.set_xlabel(x_var)

    return fig


def _plot_measure(
        results, measure, x_var, split_vars,
        kwarg_func=None, kwargs=None, ax=None):

    if ax is None:
        ax = plt.gca()

    if kwargs is None:
        kwargs = {}

    measure_data = results[split_vars + [x_var, measure]]
    grouped = measure_data.groupby(split_vars + [x_var])
    mean = grouped.mean()
    logger.info(mean)

    ci_lower = pd.Series(data=0.0, index=mean.index)
    ci_upper = pd.Series(data=0.0, index=mean.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, group in grouped:
            values = group[measure].values
            values = values[np.logical_not(np.isnan(values))]

            if len(values) > 1:
                try:
                    ci = bootstrap.ci(values)
                except:
                    ci = values[0], values[0]

            elif len(values) == 1:
                ci = values[0], values[0]
            else:
                raise Exception(
                    "No values for measure %s with "
                    "index %s." % (measure, name))

            ci_lower[name] = ci[0]
            ci_upper[name] = ci[1]

    mean['ci_lower'] = ci_lower
    mean['ci_upper'] = ci_upper

    mean = mean.reset_index()
    mean = mean[np.logical_not(np.isnan(mean[measure].values))]

    sv_series = pd.Series(
        zip(*[mean[sv] for sv in split_vars]), index=mean.index)

    for sv in sv_series.unique():
        data = mean[sv_series == sv]

        y_lower = data[measure].values - data['ci_lower'].values
        y_upper = data['ci_upper'].values - data[measure].values
        yerr = np.vstack((y_lower, y_upper))

        X = data[x_var].values
        Y = data[measure].values

        _kwargs = kwargs.copy()
        if callable(kwarg_func):
            if len(sv) == 1:
                sv = sv[0]
            _kwargs.update(kwarg_func(sv))

        ax.errorbar(X, Y, yerr=yerr, **_kwargs)

    lo_x = results[x_var].min()
    hi_x = results[x_var].max()
    xlim_lo = lo_x - 0.05 * (hi_x - lo_x)
    xlim_hi = hi_x + 0.05 * (hi_x - lo_x)
    ax.set_xlim(xlim_lo, xlim_hi)

    lo_y = results[measure].min()
    hi_y = results[measure].max()

    ylim_lo = lo_y - 0.05 * (hi_y - lo_y)
    ylim_hi = hi_y + 0.05 * (hi_y - lo_y)
    ax.set_ylim(ylim_lo, ylim_hi)

    ax.set_ylabel(measure)

    return ax


def test_single_split_var():
    # An example of using plot_measures
    epsilon = 0.5
    x_values = np.linspace(-2, 2, 10)
    results = []

    funcs = [lambda a: a**2, lambda a: a]
    func_names = ['Quadratic', 'Linear']
    n_repeats = 10
    rng = np.random.RandomState(10)

    iterator = product(
        range(n_repeats), x_values, zip(funcs, func_names))

    for i, x, (f, name) in iterator:
        y = f(x) + epsilon * rng.normal()
        y_squared = f(x)**2 + epsilon * rng.normal()
        results.append(dict(
            name=name, x=x, y=y, negative_y=-y,
            y_squared=y_squared))

    results = pd.DataFrame.from_records(results)

    def kwarg_func(split_var):
        label = "Function: %s" % split_var
        if split_var == 'Quadratic':
            return dict(label=label, linestyle='-')
        else:
            return dict(label=label, linestyle='--')

    plot_measures(
        results, measures=['y', 'negative_y', 'y_squared'],
        x_var='x', split_var='name', kwarg_func=kwarg_func)

    plt.show()


def test_multiple_split_vars():
    # An example of using plot_measures
    epsilon = [0.5, 2.0]
    x_values = np.linspace(-2, 2, 10)
    results = []

    funcs = [lambda a: a**2, lambda a: a]
    func_names = ['Quadratic', 'Linear']
    n_repeats = 10
    rng = np.random.RandomState(10)

    iterator = product(
        range(n_repeats), x_values, zip(funcs, func_names), epsilon)

    for i, x, (f, name), e in iterator:
        y = f(x) + e * rng.normal()
        y_squared = f(x)**2 + e * rng.normal()
        results.append(dict(
            name=name, x=x, y=y, negative_y=-y,
            y_squared=y_squared, epsilon=e))

    results = pd.DataFrame.from_records(results)

    def kwarg_func(split_var):
        name, epsilon = split_var

        label = "Function: %s, Noise Level: %f" % (name, epsilon)
        if name == 'Quadratic':
            return dict(label=label, linestyle='-')
        else:
            return dict(label=label, linestyle='--')

    plot_measures(
        results, measures=['y', 'negative_y', 'y_squared'],
        x_var='x', split_var=['name', 'epsilon'], kwarg_func=kwarg_func)

    plt.show()

if __name__ == "__main__":
    test_single_split_var()
    test_multiple_split_vars()
