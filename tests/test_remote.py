import unittest

from tflt.remote import EvalSpec, build_lm_eval_command, render_slurm_script


class RemoteCommandTest(unittest.TestCase):
    def test_eval_command_includes_dtype(self):
        cmd = build_lm_eval_command(
            EvalSpec(
                model="qwen3-0.6b-base",
                tasks="sciq",
                limit=1,
                loop_config=None,
                output_dir="/tmp/baseline",
                dtype="float16",
            )
        )
        self.assertIn("--dtype", cmd)
        self.assertIn("float16", cmd)

    def test_slurm_script_uses_full_gres_and_hpc2_environment(self):
        script = render_slurm_script(
            command=["python", "-m", "tflt.eval_runner", "--dtype", "float16"],
            job_name="tflt-debug",
            result_root="/remote/project/runs/debug/baseline",
            partition="debug",
            gres="gpu:a40:1",
            remote_src="/remote/project",
            hf_endpoint="https://hf-mirror.com",
            hf_home="/remote/cache/huggingface",
            transformers_cache="/remote/cache/huggingface/transformers",
            hf_datasets_cache="/remote/cache/huggingface/datasets",
        )
        self.assertIn("#SBATCH --partition=debug", script)
        self.assertIn("#SBATCH --gres=gpu:a40:1", script)
        self.assertIn("module load anaconda3 cuda/12.4 uv", script)
        self.assertIn('cd "/remote/project"', script)
        self.assertIn('. "/remote/project/.venv/bin/activate"', script)
        self.assertIn('export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"', script)
        self.assertIn(
            'export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/remote/cache/huggingface/datasets}"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
