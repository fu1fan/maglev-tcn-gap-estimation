from .testbench import build_testbench_csv


def export_quant_headers(*args, **kwargs):
    from .hpp_export import export_quant_headers as _impl

    return _impl(*args, **kwargs)


def export_quantized_pack(*args, **kwargs):
    from .quant_pow2 import export_quantized_pack as _impl

    return _impl(*args, **kwargs)


def load_stream_model(*args, **kwargs):
    from .streaming_tcn import load_stream_model as _impl

    return _impl(*args, **kwargs)


def run_stream_inference(*args, **kwargs):
    from .streaming_tcn import run_stream_inference as _impl

    return _impl(*args, **kwargs)


class StreamTCNExact:
    def __new__(cls, *args, **kwargs):
        from .streaming_tcn import StreamTCNExact as _impl

        return _impl(*args, **kwargs)


__all__ = [
    "StreamTCNExact",
    "build_testbench_csv",
    "export_quant_headers",
    "export_quantized_pack",
    "load_stream_model",
    "run_stream_inference",
]
