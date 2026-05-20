"""Download UnitCommitment.jl/MATPOWER daily case files.

The solver scripts expect files under:

    case_data/matpower/<local_case_name>/<YYYY-MM-DD>.json.gz

By default this script downloads the 365 daily files for one upstream case and
stores them under the same local case name.
"""

from __future__ import annotations

import argparse
from calendar import monthrange
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://axavier.org/UnitCommitment.jl/0.4/instances/matpower"


def _iter_daily_filenames(year: int):
    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):
            yield f"{year:04d}-{month:02d}-{day:02d}.json.gz"


def download_case_data(
    case_name: str,
    year: int = 2017,
    case_root: str | Path = "case_data/matpower",
    output_case_name: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """Download one case-year and return the local output directory."""
    local_case_name = output_case_name or case_name
    output_dir = Path(case_root) / local_case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_root = f"{base_url.rstrip('/')}/{case_name}"
    downloaded = 0
    skipped = 0
    for file_name in _iter_daily_filenames(year):
        save_path = output_dir / file_name
        if save_path.exists() and not overwrite:
            skipped += 1
            continue

        url = f"{upstream_root}/{file_name}"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        downloaded += 1
        print(f"downloaded {file_name}")

    print(
        f"done: case={case_name}, local={local_case_name}, "
        f"downloaded={downloaded}, skipped={skipped}, path={output_dir}"
    )
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", default="case118", help="Upstream MATPOWER case name.")
    parser.add_argument(
        "--output-case-name",
        default=None,
        help="Local directory name. Use this when the code uses a local alias such as case1888rte_pro.",
    )
    parser.add_argument("--year", type=int, default=2017, help="Data year to download.")
    parser.add_argument("--case-root", default="case_data/matpower", help="Local root directory for case data.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="UnitCommitment.jl MATPOWER instance base URL.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    download_case_data(
        case_name=args.case_name,
        year=args.year,
        case_root=args.case_root,
        output_case_name=args.output_case_name,
        base_url=args.base_url,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
