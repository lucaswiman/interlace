Installation
=============

Prerequisites
-------------

- Python 3.7+
- pip or another Python package manager


Install from Source
-------------------

If you're installing from the repository:

.. code-block:: bash

   cd interlace
   pip install -e .


Running Tests
-------------

To verify your installation:

.. code-block:: bash

   cd interlace
   make test

This runs the full test suite (40 tests across all modules):

- ``test_trace_markers.py`` - Synchronous trace marker tests
- ``test_async_trace_markers.py`` - Asynchronous trace marker tests
- ``test_bytecode.py`` - Bytecode instrumentation tests (experimental)
- ``test_async_bytecode.py`` - Async bytecode instrumentation tests (experimental)


Project Structure
-----------------

.. code-block:: text

   interlace/
   ├── interlace/                  # Python package
   │   ├── __init__.py
   │   ├── trace_markers.py        # Trace marker approach (recommended)
   │   ├── async_trace_markers.py  # Async trace markers
   │   ├── bytecode.py             # Bytecode instrumentation (experimental)
   │   ├── async_bytecode.py       # Async bytecode instrumentation (experimental)
   │   └── async_scheduler.py      # Async scheduling utilities
   ├── tests/                      # Test suite
   ├── docs/                       # Documentation
   └── Makefile                    # Build targets
