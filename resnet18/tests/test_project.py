"""Regression tests; all generated training artifacts stay in temporary folders."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import latency
import split
import train


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_split_does_not_modify_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.csv'
            path.write_text('path,label,frame_index\na.jpg,red,0\nb.jpg,red,1\nc.jpg,red,2\nd.jpg,red,3\n'
                            'e.jpg,brown,0\nf.jpg,brown,1\ng.jpg,brown,2\nh.jpg,brown,3\n')
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                split.create_splits(path, gap=5)
            self.assertEqual(path.read_bytes(), original)

    def test_identical_images_across_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a', 'b', 'c', 'd'):
                Image.new('RGB', (8, 8), (20, 30, 40)).save(root / f'{name}.png')
            metadata = root / 'metadata.csv'
            metadata.write_text('path,label,frame_index,group,split\n'
                                'a.png,box_merah,0,red,train\nb.png,box_cokelat,0,brown,train\n'
                                'c.png,box_merah,2,red,validation\nd.png,box_cokelat,2,brown,validation\n')
            with patch.object(train, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Identical image'):
                train.load_metadata(metadata)

    def test_frozen_backbone_and_optimizer_for_all_models(self):
        for architecture in train.ARCHITECTURES:
            for mode in train.MODES:
                with self.subTest(architecture=architecture, mode=mode):
                    model = train.make_model(architecture, mode, 2, pretrained=False)
                    train.set_training_mode(model, architecture, mode)
                    head = train.get_head(model, architecture)
                    head_ids = {id(p) for p in head.parameters()}
                    self.assertTrue(head.training)
                    self.assertTrue(all(p.requires_grad for p in head.parameters()))
                    if mode == 'feature':
                        self.assertFalse(model.training)
                        self.assertTrue(all(not p.requires_grad for p in model.parameters() if id(p) not in head_ids))
                        for module in model.modules():
                            if isinstance(module, torch.nn.BatchNorm2d):
                                self.assertFalse(module.training)
                    elif mode == 'partial':
                        self.assertFalse(model.training)
                        self.assertTrue(all(m.training for m in train.partial_backbone_modules(model, architecture)))
                        optimizer = train.make_optimizer(model, architecture, mode)
                        self.assertEqual([g['lr'] for g in optimizer.param_groups], [1e-4, 1e-3])
                        actual = [id(p) for group in optimizer.param_groups for p in group['params']]
                        self.assertEqual(len(actual), len(set(actual)))
                        self.assertEqual(set(actual), {id(p) for p in model.parameters() if p.requires_grad})
                    else:
                        self.assertTrue(model.training)
                        self.assertTrue(all(p.requires_grad for p in model.parameters()))
                    model.eval()
                    with torch.inference_mode():
                        self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 2))

    def test_training_checkpoint_report_and_latency_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(3)
            dataset = TensorDataset(torch.randn(4, 3, 224, 224, generator=generator), torch.tensor([0, 1, 0, 1]))
            loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(42))
            with patch.object(train, 'ROOT', root):
                result = train.train_experiment('resnet18', 'scratch', loader, loader,
                                               {'box_cokelat': 0, 'box_merah': 1},
                                               1, torch.device('cpu'), 42, 'test-only')
                checkpoint = root / result['checkpoint']
                self.assertTrue(checkpoint.is_file())
                with (root / 'results/scratch_history.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertGreater(float(rows[0]['epoch_seconds']), 0)
                self.assertEqual(float(rows[0]['head_lr']), 1e-3)
                self.assertEqual(float(rows[0]['val_accuracy']), result['best_val_accuracy'])
                train.update_results([result])
                train.plot_results()
                self.assertTrue((root / 'results/accuracy_per_epoch.png').is_file())
                self.assertTrue((root / 'results/scratch_curves.png').is_file())
                self.assertTrue((root / 'results/experiments.md').is_file())
                self.assertGreater(result['trainable_params'], 0)
                self.assertEqual(result['trainable_params'], result['total_params'])
                # Legacy CSVs have no LR or timing values; plotting must still work.
                history_path = root / 'results/scratch_history.csv'
                legacy_fields = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
                with history_path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=legacy_fields, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(rows)
                train.plot_results()
                image = root / 'input.png'
                Image.new('RGB', (320, 240), 'red').save(image)
                measured = latency.benchmark_checkpoint(checkpoint, image, torch.device('cpu'), 1, 2,
                                                        'resnet18', 'scratch')
                self.assertGreater(measured['median_ms'], 0)
                output = root / 'results/latency.csv'
                latency.write_result(output, measured)
                latency.write_result(output, measured)
                with output.open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)
                with self.assertRaisesRegex(ValueError, 'Expected mode'):
                    latency.benchmark_checkpoint(checkpoint, image, torch.device('cpu'), 1, 1,
                                                 'resnet18', 'feature')


if __name__ == '__main__':
    unittest.main()
