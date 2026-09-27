"""One-command Rust HTTP serving, immutable model download and diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys
from . import __version__


def main():
    parser = argparse.ArgumentParser(prog='gemmadecision', description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='Start Granian Rust HTTP serving; downloads pinned model once')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8700)
    serve.add_argument('--model-path', type=str)
    serve.add_argument('--backend', choices=['torch', 'vllm'], default='torch')
    serve.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    serve.add_argument('--strict', action='store_true', help='Singleton reference mode; slower, useful for parity')
    serve.add_argument('--offline', action='store_true')
    serve.add_argument('--max-batch-tokens', type=int, default=8192)
    serve.add_argument('--max-batch-size', type=int, default=32)
    serve.add_argument('--batch-wait-ms', type=float, default=1.0)
    serve.add_argument('--batch-requests', type=int, default=32)
    serve.add_argument('--queue-size', type=int, default=128)
    serve.add_argument('--request-timeout', type=float, default=60)
    serve.add_argument('--cache-size', type=int, default=0, help='Exact joint-score cache entries; default off')
    serve.add_argument('--gpu-memory-utilization', type=float, default=.2)
    serve.add_argument('--allow-unauthenticated', action='store_true', help='Explicitly allow non-loopback HTTP without an API key')
    download = sub.add_parser('download', help='Download immutable weights for subsequent offline use')
    download.add_argument('--output', type=Path)
    sub.add_parser('doctor', help='Print installed optional dependencies without loading a model')
    args = parser.parse_args()
    if args.command == 'download':
        from .model import download_model
        print(download_model(local_dir=args.output))
        return
    if args.command == 'doctor':
        from importlib.metadata import version, PackageNotFoundError
        result = {'gemmadecision': __version__, 'python': sys.version.split()[0]}
        for name in ['granian', 'torch', 'transformers', 'vllm', 'pydantic-ai-slim']:
            try:
                result[name] = version(name)
            except PackageNotFoundError:
                result[name] = None
        print(json.dumps(result, indent=2))
        return
    if args.host not in {'127.0.0.1', 'localhost', '::1'} and not os.getenv('GEMMADECISION_API_KEY') and not args.allow_unauthenticated:
        parser.error('Set GEMMADECISION_API_KEY for network serving, or explicitly pass --allow-unauthenticated')
    if not 1 <= args.port <= 65535 or args.max_batch_size < 1 or args.max_batch_tokens < 1 or args.queue_size < 1:
        parser.error('Port, batch and queue sizes must be positive and valid')
    if not 0 <= args.batch_wait_ms <= 100 or args.request_timeout <= 0:
        parser.error('batch-wait-ms must be 0..100 and request-timeout must be positive')
    try:
        from granian import Granian
        from granian.constants import Interfaces
    except ImportError:
        parser.error("Install the server first: pip install 'gemmadecision[serve]'")
    config = {k: v for k, v in vars(args).items() if k not in {'command', 'host', 'port', 'allow_unauthenticated'}}
    if config['model_path'] is None:
        config.pop('model_path')
    os.environ['GEMMADECISION_CONFIG'] = json.dumps(config)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    # One model process avoids accidental duplicate GPU memory allocations.
    Granian('gemmadecision.server:app', address=args.host, port=args.port,
            interface=Interfaces.ASGI, workers=1, backpressure=args.queue_size,
            log_access=False).serve()


if __name__ == '__main__':
    main()
