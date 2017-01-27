***************
Developer Guide
***************

Building the documentation
==============================
We use `the same process as nengo <https://pythonhosted.org/nengo/workflow.html>`_ to build the documentation.

Development workflow
====================
Development happens on `Github <https://github.com/nengo/nengo_mpi>`_.
Feel free to fork any of our repositories and send a pull request!
However, note that we ask contributors to sign
`a copyright assignment agreement <https://github.com/nengo/nengo/blob/master/LICENSE.rst>`_.

Code style
**********
For python code, we use the same conventions as nengo: PEP8, flake8 for checking, and numpydoc For
docstrings. See the `nengo code style <https://pythonhosted.org/nengo/workflow.html>`_ guide.

For C++ code, we roughly adhere to Google's `style guide <https://google-styleguide.googlecode.com/svn/trunk/cppguide.html>`_.

Unit testing
************
We use `PyTest <http://pytest.org/latest/>`_ to run our unit tests on `Travis-CI <https://travis-ci.com/>`_.
We run nengo's full test-suite using nengo_mpi as a back-end.  We also have a number of tests to explicitly ensure that results obtained using nengo_mpi are the same as those obtained using Nengo to a high-degree of accuracy.
Tests are run using the invocation: ::

    py.test

