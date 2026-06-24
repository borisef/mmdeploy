import os
from typing import Optional, Sequence

import onnxruntime as ort

from mmdeploy.utils import Backend
from ...base import BACKEND_WRAPPER
from ..wrapper import ORTWrapper


@BACKEND_WRAPPER.register_module(Backend.ONNXRUNTIME.value, force=True)
class ORTWrapperExtended(ORTWrapper):

    def __init__(self,
                 onnx_file: str,
                 device: str,
                 output_names: Optional[Sequence[str]] = None):
        _orig_cls = ort.SessionOptions

        def _patched():
            so = _orig_cls()
            if os.environ.get('MMDEPLOY_ORT_OPTIMIZE_ALL'):
                so.graph_optimization_level = \
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return so

        ort.SessionOptions = _patched
        try:
            super().__init__(onnx_file, device, output_names)
        finally:
            ort.SessionOptions = _orig_cls
