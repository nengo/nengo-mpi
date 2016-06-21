from __future__ import print_function
from plot import plot_measures
import os
import re
import subprocess
import argparse
from zipfile import ZipFile, ZIP_DEFLATED
from collections import namedtuple
from itertools import product
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import seaborn
seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

VERBOSE = False
MAX_PROCS = np.inf
DO_REFIMPL = True
RNG = np.random.RandomState()
maxint = np.iinfo(np.int32).max

base_dirname = 'benchmark_netfiles'


def extract_timing(text):
    """ Extract timing information from the output of nengo_mpi. """

    output = {}

    build_matches = re.findall('Building network took \d+\.\d+', text)
    if build_matches:
        assert len(build_matches) == 1
        output['build'] = float(build_matches[0].split(' ')[-1])

    load_matches = re.findall(
        'Loading network from file took \d+\.\d+', text)
    if load_matches:
        assert len(load_matches) == 1
        output['load'] = float(load_matches[0].split(' ')[-1])

    sim_matches = re.findall('Simulating \d+ steps took \d+\.\d+', text)
    if sim_matches:
        assert len(sim_matches) == 1
        output['simulate'] = float(sim_matches[0].split(' ')[-1])

    total_matches = re.findall(
        'Total time for running script was \d+\.\d+', text)
    if total_matches:
        assert len(total_matches) == 1
        output['total'] = float(total_matches[0].split(' ')[-1])

    return output

FILENAME_TEMPLATE = (
    "{name}_p_{n_procs}_{x_var_name}_{x_var_value}_r_{round}.net")


def make_filename(name, n_procs, x_var_name, x_var_value, round):
    return FILENAME_TEMPLATE.format(
        name=name, n_procs=n_procs,
        x_var_name=x_var_name,
        x_var_value=x_var_value, round=round)

FILENAME_PATTERN = "(\S+)_p_(\d+)_(\S+)_(\S+)_r_(\d+).net"


def parse_filename(filename):
    groups = re.findall(FILENAME_PATTERN, filename)

    if not len(groups) == 1:
        raise ValueError("Filename %s cannot be parsed." % filename)
    name = groups[0][0]
    n_procs = int(groups[0][1])
    x_var_name = groups[0][2]
    x_var_value = float(groups[0][3])
    round = int(groups[0][4])

    return name, n_procs, x_var_name, x_var_value, round


def execute_command(command):
    try:
        output = subprocess.check_output(
            command.split(), stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise

    if VERBOSE:
        print(output)

    return output


def basename_wo_ext(filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    return basename


def build(filename, scripts, t):
    if not filename.endswith('.zip'):
        filename += '.zip'

    if os.path.isfile(filename):
        os.remove(filename)

    results = []

    with ZipFile(filename, 'w', ZIP_DEFLATED) as zf:
        for script in scripts:
            iterator = product(
                script.n_procs, range(script.n_rounds),
                zip(script.x_var_values, script.x_var_max_procs))

            for n_procs, r, (x, max_procs) in iterator:
                if n_procs > max_procs or n_procs > MAX_PROCS:
                    continue

                seed = RNG.randint(maxint)
                network_filename = make_filename(
                    script.name, n_procs, script.x_var_name, x, r)

                command = script.command_template.format(
                    x=x, n_procs=n_procs, seed=seed)
                command = "python " + command

                if n_procs > 0:
                    command += ' --save %s' % network_filename

                    print("Building %s with command:\n%s." % (
                        network_filename, command))
                    output = execute_command(command)
                    print("Done building %s.\n" % network_filename)

                    timing = extract_timing(output)
                    try:
                        results.append([
                            network_filename, script.name,
                            script.x_var_name, x, r, n_procs,
                            timing['build'], 0.0, 0.0])
                        results.append(dict(
                            filename=network_filename, script_name=script.name,
                            x_var_name=script.x_var_name, round=r,
                            n_procs=n_procs, build_time=timing['build']))
                        results[-1][script.x_var_name] = x
                    except KeyError:
                        raise Exception(
                            'No build timing information in output.')
                    zf.write(
                        network_filename,
                        os.path.join(base_dirname, network_filename))
                    os.remove(network_filename)
                else:
                    if not DO_REFIMPL:
                        continue

                    command += ' -t %f' % t

                    print("Building and simulating non-mpi network "
                          "with command:\n%s." % command)
                    output = execute_command(command)
                    print("Done building and simulating.\n")

                    timing = extract_timing(output)
                    try:
                        results.append(dict(
                            filename=network_filename, script_name=script.name,
                            x_var_name=script.x_var_name, round=r,
                            n_procs=n_procs, build_time=timing['build'],
                            sim_time=timing['simulate'],
                            load_time=timing['load']))
                        results[-1][script.x_var_name] = x
                    except KeyError:
                        raise Exception('No timing information in output.')

        df = pd.DataFrame.from_records(results, index='filename')
        csv = df.to_csv()
        zf.writestr(os.path.join(base_dirname, 'build_results.csv'), csv)


def simulate(filename, t):
    """
    Mainly want to use this on a cluster, so this really just submits jobs,
    doesn't necessarily wait until the jobs are actually finished.

    Filename points to a file created by the build function.
    t is the length of the simulation.

    Basically need to unzip the file, run each of the .net files contained
    within using submit_job.py.

    """
    if not filename.endswith('.zip'):
        filename += '.zip'

    if not os.path.isfile(filename):
        raise Exception("File %s was not found." % filename)
    assert t >= 0, "Cannot run for a negative number of seconds %g." % t

    output_directory = basename_wo_ext(filename)
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)

    try:
        with ZipFile(filename, 'r') as zf:
            for network_filename in zf.namelist():
                if not network_filename.endswith('.net'):
                    continue
                zf.extract(network_filename)
                n_procs = parse_filename(network_filename)[1]

                command = (
                    "python ../submit_job.py {network_filename} -p {n_procs} "
                    "-t {t} -w 30:00 --dir {dir}".format(
                        network_filename=network_filename, t=t,
                        n_procs=n_procs, dir=output_directory))

                print("Submitting job for network %s with command:\n%s." % (
                    network_filename, command))

                execute_command(command)

                print("Done submitting job for %s.\n" % network_filename)

            zf.extract(os.path.join(base_dirname, 'build_results.csv'))
            shutil.copy(
                os.path.join(base_dirname, 'build_results.csv'),
                os.path.join(output_directory, 'build_results.csv'))
    finally:
        if os.path.isdir(base_dirname):
            shutil.rmtree(base_dirname)


def finalize(directory):
    """
    Scrape the result files generated by simulate for runtime information,
    store the results, create plots from the results.

    """
    build_results_fn = os.path.join(directory, 'build_results.csv')
    build_df = pd.read_csv(build_results_fn, index_col='filename')

    results = []
    for working_dir, _, filenames in os.walk(directory):
        network_filename = [f for f in filenames if f.endswith('.net')]
        output_filename = [f for f in filenames if f.endswith('.out')]
        if not (network_filename and output_filename):
            continue

        network_filename = network_filename[0]
        output_filename = output_filename[0]

        print("Scraping timings for network %s." % network_filename)
        with open(os.path.join(working_dir, output_filename), 'r') as f:
            timing = extract_timing(f.read())
        print("Done scraping timings for network %s.\n" % network_filename)

        try:
            results.append(dict(
                filename=network_filename,
                sim_time=timing['simulate'],
                load_time=timing['load']))
        except KeyError:
            raise Exception('No timing information in output.')

    df = pd.DataFrame.from_records(results, index='filename')
    df = pd.concat([build_df, df], axis=1, join_axes=[build_df.index])
    df.to_csv(os.path.join(directory, 'results.csv'))


def full(directory, scripts, t):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    results = []
    for script in scripts:
        iterator = product(
            script.n_procs, range(script.n_rounds),
            zip(script.x_var_values, script.x_var_max_procs))

        for n_procs, r, (x, max_procs) in iterator:
            if n_procs > max_procs or n_procs > MAX_PROCS:
                continue

            seed = RNG.randint(maxint)

            if n_procs <= 1:
                if not DO_REFIMPL:
                    continue

                command = (
                    "python " + script.command_template +
                    " -t {t}").format(x=x, n_procs=n_procs, t=t, seed=seed)
            else:
                command = (
                    "mpirun -np {n_procs} python -m nengo_mpi " +
                    script.command_template +
                    " -t {t}").format(x=x, n_procs=n_procs, t=t, seed=seed)

            print("Building and simulating with command:\n%s." % command)
            output = execute_command(command)
            print("Done building and simulating.\n")

            network_filename = make_filename(
                script.name, n_procs, script.x_var_name, x, r)

            timing = extract_timing(output)
            try:
                results.append(dict(
                    filename=network_filename, script_name=script.name,
                    x_var_name=script.x_var_name, round=r,
                    n_procs=n_procs, build_time=timing['build'],
                    sim_time=timing['simulate'],
                    load_time=timing['load'],
                    total_time=timing['total']))
                results[-1][script.x_var_name] = x
            except KeyError:
                raise Exception('No missing timing information in output.')

    df = pd.DataFrame.from_records(results, index='filename')
    df.to_csv(os.path.join(directory, 'results.csv'))
    plot(directory)


def plot(directory):
    df = pd.read_csv(os.path.join(directory, 'results.csv'))

    markers = MarkerStyle.filled_markers
    colors = seaborn.xkcd_rgb.values()

    def kwarg_func(n_procs):
        if n_procs == 0:
            label = "Ref Impl"
            ls = '--'
        else:
            label = "n_procs: %d" % n_procs
            ls = '-'

        marker = markers[n_procs % len(markers)]
        c = colors[hash(str(n_procs)) % len(colors)]

        return dict(label=label, linestyle=ls, c=c, marker=marker)

    names = df['script_name'].unique()
    for name in names:
        df_for_name = df[df['script_name'] == name]
        x_var_name = df_for_name['x_var_name'].unique()[0]

        measures = 'build_time sim_time load_time total_time'.split()
        measures = [m for m in measures if m in df_for_name.columns]

        for measure in measures:
            fig = plt.figure()
            plot_measures(
                df_for_name, measures=[measure], x_var=x_var_name,
                split_var='n_procs', kwarg_func=kwarg_func)
            plot_name = '%s_%s_against_%s.pdf' % (name, measure, x_var_name)
            fig.savefig(os.path.join(directory, plot_name))


def compare(dir0, dir1):
    name0 = os.path.split(dir0)[-1]
    name1 = os.path.split(dir1)[-1]
    assert name0 != name1

    df0 = pd.read_csv(os.path.join(dir0, 'results.csv'))
    df0['dataset'] = name0
    df0.reset_index()

    df1 = pd.read_csv(os.path.join(dir1, 'results.csv'))
    df1['dataset'] = name1
    df1.reset_index()

    df = df0.append(df1, ignore_index=True)

    directory = "%s_vs_%s" % (name0, name1)
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    markers = MarkerStyle.filled_markers
    colors = seaborn.xkcd_rgb.values()

    def kwarg_func(x):
        n_procs, dataset = x

        ls = '--' if dataset == name0 else '-'
        label = "%s, n_procs: %d" % (dataset, n_procs)
        marker = markers[n_procs % len(markers)]
        c = colors[hash(str(n_procs)) % len(colors)]

        return dict(label=label, linestyle=ls, c=c, marker=marker)

    names = df['script_name'].unique()
    for name in names:
        df_for_name = df[df['script_name'] == name]
        x_var_name = df_for_name['x_var_name'].unique()[0]

        measures = 'build_time sim_time load_time total_time'.split()
        measures = [m for m in measures if m in df_for_name.columns]

        for measure in measures:
            fig = plt.figure()
            plot_measures(
                df_for_name, measures=[measure], x_var=x_var_name,
                split_var=['n_procs', 'dataset'], kwarg_func=kwarg_func)
            plot_name = '%s_%s_against_%s.pdf' % (
                name, measure, x_var_name)
            fig.savefig(os.path.join(directory, plot_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'task', type=str,
        choices=['build', 'simulate', 'finalize',
                 'plot', 'compare', 'full'],
        help='The task to perform.')

    parser.add_argument(
        'filenames', nargs='+',
        help='Name of output zip file if task is "build", directory '
             'to store results if task is "full", or input zip file if '
             'task is "simulate".')

    parser.add_argument(
        '-t', type=float, default=1.0,
        help='Length of the simulation in seconds. '
             'Not used if task is "build".')

    parser.add_argument(
        '--seed', type=int, default=-1,
        help='Seed for random number generation.')

    parser.add_argument(
        '-v', action='store_true', help='Verbose output.')

    parser.add_argument(
        '--size', type=str, choices=['big', 'medium', 'small'],
        default='medium', help='Maximum size of networks to benchmark on.')

    parser.add_argument(
        '--no-ref', action='store_true',
        help='Do not run the reference implementation.')

    parser.add_argument(
        '--max-procs', type=int, default=np.inf,
        help='Maximum number of processors to use.')

    args = parser.parse_args()
    print(args)

    task = args.task
    filenames = args.filenames
    assert (
        len(filenames) == 2 if task == 'compare'
        else len(filenames) == 1)
    t = args.t
    VERBOSE = args.v
    MAX_PROCS = args.max_procs
    DO_REFIMPL = not args.no_ref
    if args.seed >= 0:
        RNG = np.random.RandomState(args.seed)

    ScriptInfo = namedtuple(
        'ScriptInfo', ['name', 'n_procs', 'command_template', 'x_var_name',
                       'x_var_values', 'x_var_max_procs', 'n_rounds'])

    if args.size == 'big':
        grid = ScriptInfo(
            name='grid',
            n_procs=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            command_template=(
                "grid.py --n-ensembles {x} -d 4 --npd 50 "
                "-p {n_procs} --seed {seed} --pfunc metis"),
            x_var_name="n_ensembles",
            x_var_values=[16, 64, 256, 1024, 4096],
            x_var_max_procs=[16, 64, 256, 1024, 4096],
            n_rounds=5)

        random_graph = ScriptInfo(
            name='random_graph',
            n_procs=[0, 1, 4, 16, 64, 246, 1024],
            command_template=(
                "random_graph.py -n 1024 -d 4 --npd 50 -q {x} "
                "-p {n_procs} --seed {seed} "
                "--pfunc metis --probes 0.2 --fake --ea"),
            x_var_name="q",
            x_var_values=np.linspace(0.01, 0.1, 10),
            x_var_max_procs=[1024]*10,
            n_rounds=5)

    elif args.size == 'small':
        grid = ScriptInfo(
            name='grid',
            n_procs=[0, 1, 2, 4, 8],
            command_template=(
                "grid.py --n-ensembles {x} -d 4 --npd 50 "
                "-p {n_procs} --seed {seed} --pfunc metis"),
            x_var_name="n_ensembles",
            x_var_values=[4, 16],
            x_var_max_procs=[4, 16],
            n_rounds=5)

        random_graph = ScriptInfo(
            name='random_graph',
            n_procs=[0, 1, 2, 4, 8],
            command_template=(
                "random_graph.py -n 8 -d 4 --npd 50 -q {x} "
                "-p {n_procs} --seed {seed} "
                "--pfunc metis --probes 0.2 --fake --ea"),
            x_var_name="q",
            x_var_values=np.linspace(0.25, 1.0, 5),
            x_var_max_procs=[8]*5,
            n_rounds=5)

    else:
        grid = ScriptInfo(
            name='grid',
            n_procs=[0, 1, 2, 4, 8, 16, 32, 64, 128],
            command_template=(
                "grid.py --n-ensembles {x} -d 4 --npd 50 "
                "-p {n_procs} --seed {seed} --pfunc metis"),
            x_var_name="n_ensembles",
            x_var_values=[16, 32, 64, 128],
            x_var_max_procs=[16, 32, 64, 128],
            n_rounds=5)

        random_graph = ScriptInfo(
            name='random_graph',
            n_procs=[0, 1, 2, 4, 8],
            command_template=(
                "random_graph.py -n 64 -d 4 --npd 50 -q {x} "
                "-p {n_procs} --seed {seed} "
                "--pfunc metis --probes 0.2 --fake --ea"),
            x_var_name="q",
            x_var_values=np.linspace(0.01, 0.1, 10),
            x_var_max_procs=[64]*10,
            n_rounds=5)

    scripts = [grid, random_graph]

    if task == 'full':
        full(filenames[0], scripts, t)
    elif task == 'build':
        build(filenames[0], scripts, t)
    elif task == 'simulate':
        simulate(filenames[0], t)
    elif task == 'finalize':
        finalize(filenames[0])
    elif task == 'plot':
        plot(filenames[0])
    elif task == 'compare':
        compare(*filenames)
    else:
        raise NotImplementedError()
