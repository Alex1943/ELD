"""Evaluation device, full-image padding and train-mode regression checks."""
import json
import contextlib
import io
import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_raw


class PairDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {'input':torch.ones(4,17,19), 'target':torch.ones(4,17,19),
                'metadata':{'red_gain':torch.ones(1)},
                'subset':'noise', 'sample_id':str(index)}


class RecordingProtocol:
    def __init__(self, device, fail=False):
        self.device = device
        self.fail = fail
        self.count = 0

    def calculate(self, pred, target, meta):
        assert pred.device == target.device == meta['red_gain'].device == self.device
        assert pred.shape == target.shape == (1,4,17,19)
        torch.testing.assert_close(pred, target)
        self.count += 1
        if self.fail:
            raise RuntimeError('metric failure')
        return [{'psnr':30., 'ssim':.9, 'mu_psnr':31., 'mu_ssim':.95}]

    def description(self):
        return {'test':'recording'}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        train_raw.torch, train_raw.tqdm = torch, tqdm

    def test_keeps_device_and_restores_training(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda:0'))
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as root:
                model = torch.nn.Identity().to(device).train()
                protocol = RecordingProtocol(device)
                elapsed = train_raw.evaluate(model, DataLoader(PairDataset(),batch_size=1),
                                             device, protocol, root, 1, 0)
                self.assertTrue(model.training)
                self.assertEqual(protocol.count, 2)
                self.assertGreater(elapsed, 0)
                with open(os.path.join(root, 'test_01_timing.json')) as handle:
                    self.assertEqual(json.load(handle)['count'], 2)

    def test_restores_mode_after_metric_error(self):
        device = torch.device('cpu')
        model = torch.nn.Identity().train()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'metric failure'):
                train_raw.evaluate(model, DataLoader(PairDataset(),batch_size=1),
                                   device, RecordingProtocol(device,fail=True), root, 1, 0)
        self.assertTrue(model.training)

    def test_duration_includes_long_runs(self):
        self.assertEqual(train_raw.format_duration(90061), '25:01:01')

    def test_results_and_snapshots_saved_with_log(self):
        import yaml
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()) as console:
            source = {'name':'test', 'device_ids':'1', 'experiment_note':'记录本次实验'}
            config_path = os.path.join(root, 'input.json')
            with open(config_path, 'w', encoding='utf-8') as handle:
                json.dump(source, handle, ensure_ascii=False)
            outdir = os.path.join(root, 'experiment')
            handlers = train_raw.configure_logging(outdir)
            try:
                resolved = dict(source, device_ids='0', global_batch_size=8)
                train_raw.save_configs(outdir, config_path, source, resolved)
                train_raw.log_message('iter=100 elapsed=00:00:10 ETA=00:10:00')
                train_raw.evaluate(torch.nn.Identity().train(), DataLoader(PairDataset(),batch_size=1),
                                   torch.device('cpu'), RecordingProtocol(torch.device('cpu')), outdir, 1, 0)
                # Another launch must retain the original snapshots and append to the log.
                train_raw.save_configs(outdir, config_path, source, dict(resolved, device_ids='2'))
            finally:
                for handler in handlers:
                    train_raw.LOGGER.removeHandler(handler)
                    handler.close()
            with open(os.path.join(outdir, 'train.log'), encoding='utf-8') as handle:
                log = handle.read()
            for text in ('iter=100', 'elapsed=', 'ETA=', '[test results]', 'noise', 'overall', 'mu-SSIM'):
                self.assertIn(text, log)
                self.assertIn(text, console.getvalue())
            with open(os.path.join(outdir, 'resolved_config.yaml'), encoding='utf-8') as handle:
                self.assertEqual(yaml.safe_load(handle)['device_ids'], '0')
            with open(os.path.join(outdir, 'source_config.yaml'), encoding='utf-8') as handle:
                self.assertEqual(yaml.safe_load(handle), source)
            with open(os.path.join(outdir, 'source_config.json'), 'rb') as handle, open(config_path, 'rb') as source_handle:
                self.assertEqual(handle.read(), source_handle.read())
            with open(os.path.join(outdir, 'test_01_summary.json')) as handle:
                report = json.load(handle)
            self.assertEqual(report['noise']['count'], 2)
            self.assertEqual(report['overall']['count'], 2)
            self.assertEqual(len([p for p in os.listdir(outdir) if p.startswith('resolved_config')]), 2)


if __name__ == '__main__':
    unittest.main()
