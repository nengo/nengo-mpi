***************************
Introduction for developers
***************************

Let's start off with some basics
in case you missed them in the README.

Developer installation
======================

If you want to change parts of Nengo,
you should do a developer installation.

.. code-block:: bash

   git clone https://github.com/nengo/nengo.git
   cd nengo
   python setup.py develop --user

If you use a ``virtualenv`` (recommended!)
you can omit the ``--user`` flag.

How to build the documentation
==============================
We use the same `process as nengo <https://pythonhosted.org/nengo/workflow.html>_` to build the documentation.
