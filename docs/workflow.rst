********************
Development workflow
********************

Development happens on `Github <https://github.com/nengo/nengo_mpi>`_.
Feel free to fork any of our repositories and send a pull request!
However, note that we ask contributors to sign
`a copyright assignment agreement <https://github.com/nengo/nengo/blob/master/LICENSE.rst>`_.

Code style
==========
For python code, we use the same conventions as nengo: PEP8, flake8 for checking, and numpydoc for docstrings. See `Nengo code style <https://pythonhosted.org/nengo/workflow.html>`_.

For C++ code, we adhere to Google's `style guide <https://google-styleguide.googlecode.com/svn/trunk/cppguide.html>`_.

Unit testing
============

We use `PyTest <http://pytest.org/latest/>`_ to run our unit tests
on `Travis-CI <https://travis-ci.com/>`_.
To ensure Python 2/3 compatibility, we test with
`Tox <https://tox.readthedocs.org/en/latest/>`_.

For more information on running tests, see the README.

Git
===

We use a pretty strict ``git`` workflow
to ensure that the history of the ``master`` branch
is clean and readable.
Every commit in the ``master`` branch should pass
unit testing, including PEP8.

Developers should never edit code on the ``master`` branch.
When changing code, create a new topic branch
that implements your new feature or fixes a bug.
When you think the branch is ready to be merged,
push it to Github and create a pull request. Ideally at this
point there would be a code review process.
