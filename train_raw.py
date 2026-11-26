"""Train the original ELD U-Net on DatasetGen single-frame RAW pairs."""
import argparse
import csv
import json
import os
import sys
import importlib.util
import platform
import time
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import PureWindowsPath
from collections import defaultdict


LOGGER = logging.getLogger('eld.raw')


def log_message(message):
    if LOGGER.handlers:
        LOGGER.info(message)
    else:
        print(message, flush=True)


def load_config(path):
    with open(path, encoding='utf-8-sig') as handle:
        if str(path).lower().endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(handle)
        return json.load(handle)


def write_yaml(path, config):
    with open(path, 'w', encoding='utf-8') as handle:
        try:
            import yaml
        except ImportError:
            # JSON is valid YAML 1.2, so JSON runs need no new dependency.
            json.dump(config, handle, ensure_ascii=False, indent=2)
        else:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def configure_logging(outdir):
    os.makedirs(outdir, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter('%(asctime)s %(message)s')
    formatter.converter = lambda seconds: time.gmtime(seconds + 8 * 3600)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(outdir, 'train.log'), mode='a', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return handlers


def save_configs(outdir, source_path, source, resolved):
    # Keep previous snapshots if the same experiment directory is reused.
    suffix = ''
    if os.path.exists(os.path.join(outdir, 'resolved_config.yaml')):
        suffix = '_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    extension = os.path.splitext(source_path)[1].lower()
    original = os.path.join(outdir, 'source_config' + suffix + extension)
    shutil.copyfile(source_path, original)
    source_yaml = os.path.join(outdir, 'source_config' + suffix + '.yaml')
    if extension != '.yaml':
        write_yaml(source_yaml, source)
    resolved_yaml = os.path.join(outdir, 'resolved_config' + suffix + '.yaml')
    write_yaml(resolved_yaml, resolved)
    log_message(f'config_original={original} config_source_yaml={source_yaml} config_resolved={resolved_yaml}')


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
        log_message('%s: %s' % (key, cfg[key]))
    required = [resolve_path(cfg['manifest'], cfg['data_root']),
                os.path.join(cfg['datasetgen_root'], 'degradation_metrics', 'protocol.py'),
                os.path.join(cfg['datasetgen_root'], 'isp', 'model.bin')]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError('Required RAW file is missing: %s' % path)
    log_message('demosaic weights: %s' % required[-1])
    return cfg


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='options/raw_eld.json')
    p.add_argument('--gpu-ids', default=None, help='physical GPU IDs, e.g. 0 or 2,3')
    p.add_argument('--check-paths', action='store_true', help='validate platform paths without importing PyTorch')
    a = p.parse_args()
    source = load_config(a.config)
    if a.check_paths:
        check_paths(source)
        return
    outdir = os.path.abspath(os.path.join(source['output_dir'], source['name']))
    handlers = configure_logging(outdir)
    try:
        log_message(f'[run started] command={sys.executable} {" ".join(sys.argv)}')
        log_message(f'experiment_note={source.get("experiment_note", "")}')
        cfg = check_paths(source)
        cfg['device_ids'] = a.gpu_ids or cfg.get('device_ids', '0')
        cfg['test_workers'] = int(cfg.get('test_workers', cfg['workers']))
        cfg['global_batch_size'] = cfg['batch_size'] * max(1, len(cfg['device_ids'].split(',')))
        cfg['experiment_dir'] = outdir
        save_configs(outdir, a.config, source, cfg)
        run_training(a, cfg, outdir)
    except Exception:
        LOGGER.exception('Training failed')
        raise
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler)
            handler.close()


def run_training(a, cfg, outdir):
    gpu_ids = a.gpu_ids or cfg.get('device_ids', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    global torch, tqdm
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from dataset.raw_manifest import RawManifestDataset
    # Load the original architecture without executing models/__init__.py,
    # which eagerly imports the SID/rawpy pipeline and ELD's util package.
    spec = importlib.util.spec_from_file_location(
        'eld_raw_unet', os.path.join(os.path.dirname(__file__), 'models', 'arch', 'Unet.py'))
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = [x.strip() for x in gpu_ids.split(',') if x.strip()]
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed']) if torch.cuda.is_available() else None
    ds = RawManifestDataset(cfg['data_root'], cfg['manifest'], 'train', cfg['patch_size'], cfg['subsets'])
    test = RawManifestDataset(cfg['data_root'], cfg['manifest'], 'test', 0, cfg['subsets'])
    loader = DataLoader(ds, batch_size=cfg['batch_size']*max(1,len(ids)), shuffle=True,
                        num_workers=cfg['workers'], pin_memory=device.type=='cuda', drop_last=True)
    if not len(loader):
        raise ValueError('Training dataset is smaller than the global batch with drop_last=True')
    test_workers = int(cfg.get('test_workers', cfg['workers']))
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=test_workers,
                             pin_memory=device.type == 'cuda', persistent_workers=test_workers > 0)
    log_message(f'train={len(ds)} test={len(test)} global_batch={loader.batch_size} '
                f'test_workers={test_workers} network_device={device} ISP_device={device}')
    model = arch.UNetSeeInDark(4,4).to(device)
    if len(ids)>1 and device.type=='cuda': model = torch.nn.DataParallel(model, device_ids=list(range(len(ids))))
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9,0.999))
    # DatasetGen RawMetricProtocol is the official ISP/metric authority.
    metric_root = os.path.abspath(cfg['datasetgen_root'])
    sys.path.insert(0, metric_root)
    from degradation_metrics.protocol import RawMetricProtocol, mu_tonemap
    protocol = RawMetricProtocol(hdr_scale=16, mu=5000, demosaic='net')
    total, checkpoints = int(cfg['total_iter']), int(cfg['checkpoints'])
    if not 1 <= checkpoints <= total:
        raise ValueError('Require 1 <= checkpoints <= total_iter')
    save_steps = {(k * total + checkpoints - 1) // checkpoints: k for k in range(1, checkpoints + 1)}
    step = 0; epoch = 0
    started = time.perf_counter()
    segment_started = started
    segment_step = 0
    test_seconds = []
    train_seconds = 0.0
    while step < total:
        epoch += 1
        model.train()
        for batch in loader:
            step += 1
            x, y = batch['input'].to(device), batch['target'].to(device)
            optim.zero_grad(set_to_none=True); pred = model(x)
            loss = torch.nn.functional.l1_loss(
                mu_tonemap(torch.clamp(pred / 16.0, min=0.0)),
                mu_tonemap(torch.clamp(y / 16.0, 0.0, 1.0)))
            loss.backward(); optim.step()
            if step % cfg['log_interval'] == 0 or step == total:
                synchronize(device)
                now = time.perf_counter()
                segment_elapsed = now - segment_started
                seconds_per_iter = segment_elapsed / (step - segment_step)
                eta_train = seconds_per_iter * (total - step)
                pending_tests = sum(s >= step for s in save_steps)
                eta = eta_train + (sum(test_seconds) / len(test_seconds) * pending_tests if test_seconds else 0)
                eta_scope = 'train+test_estimate' if test_seconds else 'train_only(test_time_pending)'
                log_message(f'iter={step}/{total} l1={loss.item():.6f} '
                      f'elapsed={format_duration(now-started)} ETA={format_duration(eta)} '
                      f'ETA_scope={eta_scope} samples/s={loader.batch_size/seconds_per_iter:.2f}')
                train_seconds += segment_elapsed
                segment_started = now
                segment_step = step
            if step in save_steps:
                k = save_steps[step]
                synchronize(device)
                train_seconds += time.perf_counter() - segment_started
                path = os.path.join(outdir, f'model_{k:02d}_iter_{step:08d}.pth')
                state = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
                torch.save({'iteration':step,'model':state,'optimizer':optim.state_dict()},path)
                log_message(f'[checkpoint saved] iter={step}/{total} path={path}')
                test_seconds.append(evaluate(model, test_loader, device, protocol, outdir, k, cfg['save_image_interval']))
                segment_started = time.perf_counter()
                segment_step = step
            if step >= total: break
    log_message(f'Training finished: elapsed={format_duration(time.perf_counter()-started)} '
                f'train_time={format_duration(train_seconds)} test_time={format_duration(sum(test_seconds))}')


def evaluate(model, loader, device, protocol, outdir, checkpoint, save_interval):
    from PIL import Image
    was_training = model.training
    model.eval(); rows=[]; sums=defaultdict(list)
    eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    synchronize(device)
    started = time.perf_counter()
    last = started
    timings = defaultdict(float)
    try:
        with torch.no_grad():
            progress = tqdm(loader, total=len(loader.dataset), desc=f'test checkpoint {checkpoint:02d}', unit='img')
            for i, batch in enumerate(progress):
                now = time.perf_counter()
                timings['data_wait'] += now - last
                x=batch['input'].to(device, non_blocking=True)
                y=batch['target'].to(device, non_blocking=True)
                meta={k:v.to(device, non_blocking=True) for k,v in batch['metadata'].items()}
                h,w=x.shape[-2:]; ph=(-h)%16; pw=(-w)%16
                if ph or pw: x=torch.nn.functional.pad(x,(0,pw,0,ph),mode='replicate')
                pred=eval_model(x)[...,:h,:w]
                synchronize(device)
                inferred = time.perf_counter()
                timings['transfer_and_network'] += inferred - now
                # Keep both RAW tensors and metadata on GPU for the learned ISP.
                # DatasetGen transfers only the quantized RGB views to CPU for skimage metrics.
                scores=protocol.calculate(pred,y,meta)[0]
                synchronize(device)
                measured = time.perf_counter()
                timings['ISP_and_metrics'] += measured - inferred
                subset, sample_id = batch['subset'][0], batch['sample_id'][0]
                for k,v in scores.items():
                    if k in ('psnr','ssim','mu_psnr','mu_ssim'): sums[(subset,k)].append(v)
                rows.append({'sample_id':sample_id,'subset':subset,**scores})
                if save_interval and i % save_interval == 0:
                    rgb=protocol.raw_to_linear_rgb(pred,meta)[0]
                    from degradation_metrics.protocol import mu_tonemap, quantize_uint16
                    vis=quantize_uint16(mu_tonemap(rgb,protocol.mu))
                    folder=os.path.join(outdir,f'test_{checkpoint:02d}',subset); os.makedirs(folder,exist_ok=True)
                    Image.fromarray((vis >> 8).astype('uint8')).save(os.path.join(folder,sample_id+'.png'))
                timings['save_and_bookkeeping'] += time.perf_counter() - measured
                last = time.perf_counter()
    finally:
        model.train(was_training)
    csv_path=os.path.join(outdir,f'test_{checkpoint:02d}.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={subset:{metric:sum(vals)/len(vals) for (s,metric),vals in sums.items() if s==subset}
             for subset in sorted({r['subset'] for r in rows})}
    summary['overall']={m:sum(r[m] for r in rows)/len(rows) for m in ('psnr','ssim','mu_psnr','mu_ssim')}
    for subset, metrics in summary.items():
        metrics['count'] = len(rows) if subset == 'overall' else sum(r['subset'] == subset for r in rows)
    with open(os.path.join(outdir,f'test_{checkpoint:02d}_summary.json'),'w') as f: json.dump(summary,f,indent=2)
    log_message(f'[test results] checkpoint={checkpoint:02d} split=test')
    log_message(f'{"subset":<22} {"count":>7} {"PSNR":>12} {"SSIM":>12} {"mu-PSNR":>12} {"mu-SSIM":>12}')
    for subset, metrics in summary.items():
        log_message(f'{subset:<22} {metrics["count"]:>7d} {metrics["psnr"]:>12.6f} '
                    f'{metrics["ssim"]:>12.6f} {metrics["mu_psnr"]:>12.6f} {metrics["mu_ssim"]:>12.6f}')
    elapsed = time.perf_counter() - started
    timing_report = {'count':len(rows), 'elapsed_seconds':elapsed, 'samples_per_second':len(rows)/elapsed,
                     'network_device':str(device), 'ISP_device':str(device),
                     'seconds_by_stage':dict(timings), 'protocol':protocol.description()}
    with open(os.path.join(outdir,f'test_{checkpoint:02d}_timing.json'),'w') as f:
        json.dump(timing_report,f,indent=2)
    log_message(f'[test finished] count={len(rows)} elapsed={format_duration(elapsed)} '
                f'samples/s={len(rows)/elapsed:.2f} stage_seconds={dict(timings)}')
    return elapsed


if __name__=='__main__': main()
