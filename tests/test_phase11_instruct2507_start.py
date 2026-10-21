import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/loopscope'))
import phase11_instruct2507_start as start


class StartTests(unittest.TestCase):
    def test_two_node_barriers_precede_gold_scoring(self):
        for rank in (0, 8):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def write(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                write(root / 'receipt.json', {'modelTrainingJobId': 1})
                write(root / 'experiment.json', {'source_commit': 'source', 'model_mount': '/model',
                    'source_csv_mount': '/existing-author.csv', 'model_provenance': {}})
                with tarfile.open(root / 'runtime.tar.gz', 'w:gz') as archive:
                    data = json.dumps({'source_commit': 'source'}).encode()
                    item = tarfile.TarInfo('bundle-info.json')
                    item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
                write(root / 'plan-synthetic.json', {'tasks': [{'jobs': [{'run_root': str(root / 'synthetic-summary')}]}]})
                write(root / 'synthetic-summary/summary.json', {'status': 'RAW_COMPLETE', 'count': 1})
                commands, barriers = [], []
                def run(command, deadline, **kwargs):
                    commands.append(command)
                    if 'analyze_phase11_instruct2507.py' in ' '.join(command):
                        self.assertIn('formal node completion', barriers)
                def wait(path, deadline, description):
                    barriers.append(description)
                    if description == 'both-node synthetic validation':
                        write(path, {'status': 'RUNTIME_SMOKE_COMPLETE'})
                with patch.dict(start.os.environ, {'LOCAL_RANK': '0', 'RANK': str(rank), 'WORLD_SIZE': '16'}, clear=True), \
                        patch.object(sys, 'argv', ['start', '--root', str(root)]), \
                        patch.object(start.importlib.metadata, 'distributions', return_value=[]), \
                        patch.object(start, 'run_step', side_effect=run), \
                        patch.object(start, 'wait_file', side_effect=wait):
                    start.main()
                launch = [c for c in commands if 'phase11_instruct2507_launch.py' in ' '.join(c)]
                self.assertEqual(len(launch), 2)
                if rank == 0:
                    self.assertEqual(barriers.count('synthetic node completion'), 2)
                    self.assertEqual(barriers.count('formal node completion'), 2)
                    self.assertEqual(json.loads((root / 'launch/formal-attempt1/runtime-smoke.json').read_text())['workers'], 16)
                    self.assertEqual(len([c for c in commands if '--gold' in c]), 1)
                else:
                    self.assertIn('rank-zero closure and scoring', barriers)
                    self.assertFalse(any('--gold' in c or '--source-csv' in c for c in commands))


if __name__ == '__main__':
    unittest.main()
