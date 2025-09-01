import asyncio
from pathlib import Path

from golden_path_strands import GoldenPathStrands


def test_complete_pipeline(tmp_path: Path) -> None:
    gps = GoldenPathStrands(config={"data_dir": str(tmp_path)})
    dataset_path = asyncio.run(gps.run_complete_pipeline(["test task"]))
    assert Path(dataset_path).exists()
