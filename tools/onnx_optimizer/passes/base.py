"""The uniform Pass interface.

A pass mutates a ModelProto in place and returns the number of graph changes
it made (0 = graph untouched). Passes must be:

  * idempotent to convergence -- the engine reruns the pipeline until a full
    sweep reports zero changes;
  * safe to interrupt -- the engine snapshots the model before every pass and
    auto-reverts the pass if the bit-exact gate fails afterwards, so a buggy
    pass can never silently poison the output;
  * strictly structural unless `lossy = True` -- a lossy pass may change float
    values (it only ever runs under --allow-lossy, where the byte gate is
    replaced by a tolerance gate).
"""


class Pass:
    name = ""           # CLI identifier (kebab-case)
    description = ""    # one line for --list-passes and the report
    lossy = False       # True = not bit-exact; gated behind --allow-lossy

    def run(self, model):
        """Apply to `model` in place; return the number of changes."""
        raise NotImplementedError
