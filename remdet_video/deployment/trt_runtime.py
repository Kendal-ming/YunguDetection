"""Minimal TensorRT 10 runtime using CUDA Runtime through ctypes.

This deliberately avoids PyTorch, MMCV, PyCUDA and cuda-python so the Jetson
deployment only needs the TensorRT, NumPy and CUDA packages supplied by
JetPack.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt


class CudaRuntime:
    """Small checked wrapper around the CUDA Runtime API."""

    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        errors = []
        for library in ('libcudart.so', 'libcudart.so.13'):
            try:
                self.library = ctypes.CDLL(library)
                break
            except OSError as error:
                errors.append(str(error))
        else:
            raise RuntimeError(
                'Could not load CUDA Runtime (libcudart): ' + '; '.join(errors))

        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self.library.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaMalloc.restype = ctypes.c_int
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaFree.restype = ctypes.c_int
        self.library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.cudaMemcpyAsync.restype = ctypes.c_int
        self.library.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)]
        self.library.cudaStreamCreate.restype = ctypes.c_int
        self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamDestroy.restype = ctypes.c_int
        self.library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamSynchronize.restype = ctypes.c_int

    def check(self, error: int, operation: str) -> None:
        if error == 0:
            return
        message = self.library.cudaGetErrorString(error)
        decoded = message.decode('utf-8') if message else f'CUDA error {error}'
        raise RuntimeError(f'{operation} failed: {decoded}')

    def malloc(self, size_bytes: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(
            self.library.cudaMalloc(ctypes.byref(pointer), size_bytes),
            'cudaMalloc')
        if pointer.value is None:
            raise RuntimeError('cudaMalloc returned a null pointer.')
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self.check(
            self.library.cudaFree(ctypes.c_void_p(pointer)), 'cudaFree')

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self.check(
            self.library.cudaStreamCreate(ctypes.byref(stream)),
            'cudaStreamCreate')
        if stream.value is None:
            raise RuntimeError('cudaStreamCreate returned a null stream.')
        return int(stream.value)

    def destroy_stream(self, stream: int) -> None:
        self.check(
            self.library.cudaStreamDestroy(ctypes.c_void_p(stream)),
            'cudaStreamDestroy')

    def copy_host_to_device_async(
        self, device_pointer: int, host: np.ndarray, stream: int
    ) -> None:
        self.check(
            self.library.cudaMemcpyAsync(
                ctypes.c_void_p(device_pointer),
                ctypes.c_void_p(host.ctypes.data),
                host.nbytes,
                self.HOST_TO_DEVICE,
                ctypes.c_void_p(stream),
            ),
            'cudaMemcpyAsync(H2D)',
        )

    def copy_device_to_host_async(
        self, host: np.ndarray, device_pointer: int, stream: int
    ) -> None:
        self.check(
            self.library.cudaMemcpyAsync(
                ctypes.c_void_p(host.ctypes.data),
                ctypes.c_void_p(device_pointer),
                host.nbytes,
                self.DEVICE_TO_HOST,
                ctypes.c_void_p(stream),
            ),
            'cudaMemcpyAsync(D2H)',
        )

    def synchronize(self, stream: int) -> None:
        self.check(
            self.library.cudaStreamSynchronize(ctypes.c_void_p(stream)),
            'cudaStreamSynchronize')


class TensorRTRunner:
    """Load one static TensorRT engine and execute NumPy inputs."""

    def __init__(
        self,
        engine_path: str | Path,
        logger_severity: trt.Logger.Severity = trt.Logger.WARNING,
    ) -> None:
        self.engine_path = Path(engine_path).resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)

        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f'Could not deserialize {self.engine_path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create a TensorRT execution context.')

        self.cuda = CudaRuntime()
        self.stream = self.cuda.create_stream()
        self.device_pointers: dict[str, int] = {}
        self.host_outputs: dict[str, np.ndarray] = {}
        self.input_contract: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self.output_contract: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        try:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(int(value) for value in
                              self.context.get_tensor_shape(name))
                if any(value < 0 for value in shape):
                    raise ValueError(
                        f'Dynamic tensor {name}={shape} is not supported by '
                        'this fixed-shape runner.')
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                size_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                pointer = self.cuda.malloc(size_bytes)
                self.device_pointers[name] = pointer
                if not self.context.set_tensor_address(name, pointer):
                    raise RuntimeError(f'Could not bind TensorRT tensor {name}.')

                contract = (shape, dtype)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_contract[name] = contract
                else:
                    self.output_contract[name] = contract
                    self.host_outputs[name] = np.empty(shape, dtype=dtype)
        except Exception:
            self.close()
            raise

    def io_contract(self) -> dict:
        return {
            'inputs': {
                name: {'shape': list(shape), 'dtype': str(dtype)}
                for name, (shape, dtype) in self.input_contract.items()
            },
            'outputs': {
                name: {'shape': list(shape), 'dtype': str(dtype)}
                for name, (shape, dtype) in self.output_contract.items()
            },
        }

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(inputs) != set(self.input_contract):
            raise ValueError(
                f'Expected inputs {sorted(self.input_contract)}, got '
                f'{sorted(inputs)}.')

        for name, (shape, dtype) in self.input_contract.items():
            array = np.ascontiguousarray(inputs[name], dtype=dtype)
            if array.shape != shape:
                raise ValueError(f'Expected {name} shape {shape}, got {array.shape}.')
            self.cuda.copy_host_to_device_async(
                self.device_pointers[name], array, self.stream)

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError('TensorRT execute_async_v3 returned False.')

        for name, output in self.host_outputs.items():
            self.cuda.copy_device_to_host_async(
                output, self.device_pointers[name], self.stream)
        self.cuda.synchronize(self.stream)
        return {name: output.copy() for name, output in self.host_outputs.items()}

    def close(self) -> None:
        pointers = getattr(self, 'device_pointers', {})
        cuda = getattr(self, 'cuda', None)
        if cuda is not None:
            for pointer in pointers.values():
                cuda.free(pointer)
            pointers.clear()
            stream = getattr(self, 'stream', None)
            if stream is not None:
                cuda.destroy_stream(stream)
                self.stream = None

    def __enter__(self) -> 'TensorRTRunner':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
