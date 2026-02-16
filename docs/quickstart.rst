Quick Start
===========

This guide will get you up and running with Interlace using the recommended **Trace Markers** approach.


Basic Example: Detecting a Race Condition
------------------------------------------

Let's create a simple example that demonstrates a race condition in a counter:

.. code-block:: python

   from interlace.trace_markers import Schedule, Step, TraceExecutor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           # interlace: after_read
           temp = self.value
           temp += 1
           # interlace: before_write
           self.value = temp

   # Create a counter
   counter = Counter()

   # Define an interleaving that triggers the race condition
   # Both threads read before either writes, causing lost updates
   schedule = Schedule([
       Step("thread1", "after_read"),    # T1 reads 0
       Step("thread2", "after_read"),    # T2 reads 0 (both see same value!)
       Step("thread1", "before_write"),  # T1 writes 1
       Step("thread2", "before_write"),  # T2 writes 1 (overwrites T1's update!)
   ])

   # Execute with controlled interleaving
   executor = TraceExecutor(schedule)
   executor.run("thread1", lambda: counter.increment())
   executor.run("thread2", lambda: counter.increment())
   executor.wait(timeout=5.0)

   # Verify the race condition occurred
   assert counter.value == 1, f"Expected 1 (race condition), got {counter.value}"
   print("Race condition detected!")


Understanding Trace Markers
----------------------------

Trace markers are simple comments that tell Interlace where synchronization points are:

.. code-block:: python

   def increment(self):
       # interlace: after_read
       temp = self.value
       temp += 1
       # interlace: before_write
       self.value = temp

Two placement styles are supported:

1. **On empty lines** (recommended for clarity):

   .. code-block:: python

       def increment(self):
           # interlace: after_read
           temp = self.value
           temp += 1
           # interlace: before_write
           self.value = temp

2. **Inline with code** (compact style):

   .. code-block:: python

       def increment(self):
           temp = self.value  # interlace: after_read
           temp += 1
           self.value = temp  # interlace: before_write


Creating Schedules
-------------------

A schedule defines the execution order of marked synchronization points:

.. code-block:: python

   from interlace.trace_markers import Schedule, Step

   schedule = Schedule([
       Step("thread1", "marker_name_1"),
       Step("thread1", "marker_name_2"),
       Step("thread2", "marker_name_1"),
       Step("thread2", "marker_name_2"),
   ])

Each ``Step`` specifies:

- The thread/task name
- The marker name to execute at that step


Running with Controlled Interleaving
-------------------------------------

Execute your code with a specific schedule:

.. code-block:: python

   from interlace.trace_markers import TraceExecutor

   executor = TraceExecutor(schedule)

   # Register tasks to run
   executor.run("thread1", task_function_1)
   executor.run("thread2", task_function_2)

   # Wait for completion
   executor.wait(timeout=5.0)


Async Support
-------------

.. warning::

   **WIP**: Async trace marker syntax and semantics are still being finalized.
   See :doc:`future_work` for planned improvements to async marking syntax.


Next Steps
----------

- Read :doc:`approaches` for detailed information about trace markers and other approaches
- See :doc:`examples` for more complete examples
- Check the :doc:`api_reference` for detailed API documentation
