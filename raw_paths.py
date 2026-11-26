"""Platform-specific RAW paths, independent of training dependencies."""
import os
import platform
from pathlib import PureWindowsPath


def resolve_path(value, base, system=None):
    system = system or platform.system()
    if system == 'Linux' and PureWindowsPath(str(value)).drive:
        raise ValueError('Windows path cannot be used on Linux: %s' % value)
    if system == 'Windows' and str(value).startswith('/mnt/'):
        raise ValueError('Linux server path cannot be used on Windows: %s' % value)
    value = os.path.expanduser(str(value)).replace('\\', '/')
    return os.path.normpath(value if os.path.isabs(value) else os.path.join(base, value))


def platform_root(cfg, key, system=None):
    system = system or platform.system()
    value = cfg[key]
    if isinstance(value, dict):
        if system not in ('Windows', 'Linux'):
            raise ValueError('Unsupported platform: %s' % system)
        value = value[system.lower()]
    return resolve_path(value, os.path.dirname(os.path.abspath(__file__)), system)


def check_paths(cfg):
    cfg = dict(cfg)
    for key in ('data_root', 'datasetgen_root'):
        cfg[key] = platform_root(cfg, key)
        print('%s: %s' % (key, cfg[key]), flush=True)
    required = [resolve_path(cfg['manifest'], cfg['data_root']),
                os.path.join(cfg['datasetgen_root'], 'degradation_metrics', 'protocol.py'),
                os.path.join(cfg['datasetgen_root'], 'isp', 'model.bin')]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError('Required RAW file is missing: %s' % path)
    print('demosaic weights: %s' % required[-1], flush=True)
    return cfg
