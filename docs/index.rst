Qwen3 Circuit Reproduction — documentation
==========================================

Mechanistic interpretability of Qwen3-4B (base) with per-layer transcoders:
reproducing and extending the circuit-tracing programme of *On the Biology of a
Large Language Model* on an open model.

For the scientific narrative, installation, and end-to-end usage see the
project ``README.md``. This site documents the reusable ``src`` library.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Pipeline overview
-----------------

The pipeline is a sequence of numbered, single-purpose scripts in ``scripts/``:

* **GPU capture** — baseline (``02``), transcoder features (``04``), attribution
  graph (``06``), field dumps (``119``), realized-write dumps (``154``),
  steering sweeps (``85``, ``122``).
* **CPU analysis** — causal edges (``08``), path validation (``14``), geometry
  (``axis_geometry_by_layer``), MLP decomposition (``155``), active-span
  (``110``), null calibration (``118``), clustering (``22``–``29``), whitening
  (``133``).
* **Figures** — ``plot_*.py`` consume the committed ``results/`` CSVs.

Each ``results/`` subfolder carries a ``README.md`` mapping every file to the
report table or figure it supports.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
