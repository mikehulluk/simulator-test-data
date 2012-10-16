simulator-test-data
===================

Test data for validating neuronal simulators

'waf generate' - generate all the data by running the simulations.
'waf compare' - look at the different output traces
'waf cleanup' - cleans all the generated files


By default, this will test all simulators against all scenarios. This can by
limited by using environmental variables to choose the simulator and the
scenarios, for example:

.. code-block:: verbatim

    export STD_SIMS='morphforge;NEURON';
    export STD_SCENS='022; 5??; 62[12]';
    export STD_SHORT='TRUE';

The last option, STD_SHORT is for creating short runs for syntax checking,
since runs can take a long time.



# Waf lets you stack up actions:
'waf cleanup generate compare'
