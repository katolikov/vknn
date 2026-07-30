"""`python -m onnx_optimizer` == `python -m onnx_optimizer.optimize`."""
import sys

from onnx_optimizer.optimize import main

sys.exit(main())
