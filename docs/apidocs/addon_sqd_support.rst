.. _fulqrum-addon-sqd-support:

.. currentmodule:: fulqrum.addon_sqd_support

SQD addon support
=================

This module builds the ``sci_solver`` that
:func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` expects, so that
it is straightforward for a user to leverage Fulqrum with the addon's configuration
recovery loop.


SCI solver
----------

.. autosummary::
   :toctree: ../stubs/

   sci_solver_from_integrals
