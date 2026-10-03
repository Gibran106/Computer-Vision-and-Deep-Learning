"""Protect latency comparisons against mixed devices/settings and stale rows."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_latency


def measurement(mode='feature', **changes):
    row = dict(architecture='resnet18', mode=mode, device='cuda', device_name='GPU',
               cpu_threads='1', torch_version='2', input_size='224', batch_size='1',
               warmup_iterations='20', measured_iterations='100', image_sha256='same-image',
               mean_ms='2', median_ms='2', p95_ms='3', image='missing.jpg',
               measured_at_utc='2026-10-03T14:00:00+00:00')
    row.update(changes)
    return row


class LatencyPlotTests(unittest.TestCase):
    def test_devices_and_settings_are_never_combined(self):
        rows = [measurement(mode) for mode in plot_latency.MODES]
        rows += [measurement('feature', device='cpu', measured_at_utc='2026-10-04T14:00:00+00:00'),
                 measurement('partial', input_size='128', measured_at_utc='2026-10-04T14:00:00+00:00'),
                 measurement('scratch', image_sha256='different-image', measured_at_utc='2026-10-04T14:00:00+00:00')]
        selected = plot_latency.select_comparable(rows)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(r['device'] == 'cuda' and r['input_size'] == '224' and r['image_sha256'] == 'same-image' for r in selected))
        self.assertEqual(len(plot_latency.select_comparable(rows, 'cpu')), 1)

    def test_latest_timestamp_wins_even_when_csv_is_out_of_order(self):
        newest = measurement(median_ms='4', measured_at_utc='2026-10-04T14:00:00+00:00')
        selected = plot_latency.select_comparable([newest, measurement()])
        self.assertEqual(selected[0]['median_ms'], '4')
        with self.assertRaisesRegex(ValueError, 'No measured latency'):
            plot_latency.select_comparable([newest], 'cpu')

    def test_non_finite_measurements_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'resnet18'
            (folder / 'results').mkdir(parents=True)
            row = measurement(median_ms='nan')
            with (folder / 'results/latency.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, 'Invalid median_ms'):
                plot_latency.read_measurements(folder)


if __name__ == '__main__':
    unittest.main()
