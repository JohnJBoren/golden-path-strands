"""Minimal usage example for the Golden Path Strands framework."""
import asyncio

from golden_path_strands import GoldenPathStrands


async def main() -> None:
    gps = GoldenPathStrands()
    dataset_path = await gps.run_complete_pipeline(["example task"])
    print(f"Dataset written to {dataset_path}")


if __name__ == "__main__":
    asyncio.run(main())
