"""Convert a trusted MMEngine checkpoint into a tensor-only state_dict file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument(
        '--weights-key', default='state_dict',
        help='Checkpoint mapping to export, e.g. state_dict or ema_state_dict')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or args.weights_key not in checkpoint:
        raise TypeError(
            f'Expected an MMEngine checkpoint with {args.weights_key!r}')
    state_dict = checkpoint[args.weights_key]
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError('state_dict must be a non-empty mapping')
    if args.weights_key == 'ema_state_dict':
        # MMEngine's EMAHook stores an internal ``steps`` counter and prefixes
        # the averaged model weights with ``module.``. Convert that hook state
        # into the exact key format expected by a normal detector checkpoint.
        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
            if key != 'steps'
        }
    invalid = [key for key, value in state_dict.items()
               if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f'Non-tensor state_dict entries: {invalid[:5]}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state_dict}, destination)
    parameter_values = sum(value.numel() for value in state_dict.values())
    print(f'source_keys={sorted(checkpoint)}')
    print(f'exported_key={args.weights_key}')
    print(f'tensors={len(state_dict)} tensor_values={parameter_values}')
    print(f'wrote={destination} bytes={destination.stat().st_size}')


if __name__ == '__main__':
    main()
