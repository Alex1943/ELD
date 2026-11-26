"""Train the original ELD U-Net on DatasetGen single-frame RAW pairs."""
import argparse
import csv
import json
import os
import sys
import importlib.util
from collections import defaultdict

from raw_paths import check_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='options/raw_eld.json')
    p.add_argument('--gpu-ids', default=None, help='physical GPU IDs, e.g. 0 or 2,3')
    p.add_argument('--check-paths', action='store_true', help='validate platform paths without importing PyTorch')
    a = p.parse_args()
    with open(a.config, encoding='utf-8') as handle:
        cfg = check_paths(json.load(handle))
    if a.check_paths:
        return
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
    model = arch.UNetSeeInDark(4,4).to(device)
    if len(ids)>1 and device.type=='cuda': model = torch.nn.DataParallel(model, device_ids=list(range(len(ids))))
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9,0.999))
    outdir = os.path.join(cfg['output_dir'], cfg['name']); os.makedirs(outdir, exist_ok=True)
    # DatasetGen RawMetricProtocol is the official ISP/metric authority.
    metric_root = os.path.abspath(cfg['datasetgen_root'])
    sys.path.insert(0, metric_root)
    from degradation_metrics.protocol import RawMetricProtocol, mu_tonemap
    protocol = RawMetricProtocol(hdr_scale=16, mu=5000, demosaic='net')
    total, checkpoints = int(cfg['total_iter']), int(cfg['checkpoints'])
    interval = max(1, total // checkpoints); step = 0; epoch = 0
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
            if step % cfg['log_interval'] == 0: print(f'iter={step}/{total} l1={loss.item():.6f}')
            if step % interval == 0 or step == total:
                k = min(checkpoints, round(step / total * checkpoints))
                path = os.path.join(outdir, f'model_{k:02d}_iter_{step:08d}.pth')
                state = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
                torch.save({'iteration':step,'model':state,'optimizer':optim.state_dict()},path)
                evaluate(model, test, device, protocol, outdir, k, cfg['save_image_interval'])
            if step >= total: break


def evaluate(model, dataset, device, protocol, outdir, checkpoint, save_interval):
    from PIL import Image
    model.eval(); rows=[]; sums=defaultdict(list)
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f'test checkpoint {checkpoint:02d}'):
            item=dataset[i]; x=item['input'].unsqueeze(0).to(device); y=item['target'].unsqueeze(0)
            h,w=x.shape[-2:]; ph=(-h)%16; pw=(-w)%16
            if ph or pw: x=torch.nn.functional.pad(x,(0,pw,0,ph),mode='replicate')
            pred=model(x)[...,:h,:w].cpu(); meta={k:v.unsqueeze(0) for k,v in item['metadata'].items()}
            scores=protocol.calculate(pred,y,meta)[0]
            for k,v in scores.items():
                if k in ('psnr','ssim','mu_psnr','mu_ssim'): sums[(item['subset'],k)].append(v)
            rows.append({'sample_id':item['sample_id'],'subset':item['subset'],**scores})
            if save_interval and i % save_interval == 0:
                rgb=protocol.raw_to_linear_rgb(pred,meta)[0]
                from degradation_metrics.protocol import mu_tonemap, quantize_uint16
                vis=quantize_uint16(mu_tonemap(rgb,5000))
                folder=os.path.join(outdir,f'test_{checkpoint:02d}',item['subset']); os.makedirs(folder,exist_ok=True)
                Image.fromarray((vis[...,::-1] >> 8).astype('uint8')).save(os.path.join(folder,item['sample_id']+'.png'))
    csv_path=os.path.join(outdir,f'test_{checkpoint:02d}.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={subset:{metric:sum(vals)/len(vals) for (s,metric),vals in sums.items() if s==subset}
             for subset in sorted({r['subset'] for r in rows})}
    summary['overall']={m:sum(r[m] for r in rows)/len(rows) for m in ('psnr','ssim','mu_psnr','mu_ssim')}
    with open(os.path.join(outdir,f'test_{checkpoint:02d}_summary.json'),'w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
