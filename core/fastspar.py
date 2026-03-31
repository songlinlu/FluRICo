from __future__ import annotations

import concurrent.futures
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


def _supports_threads(fastspar_bin: str) -> bool:
    try:
        result = subprocess.run(
            [fastspar_bin, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        return "--threads" in text
    except Exception:
        return False


def _run_one(
    *,
    fastspar_bin: str,
    otu_file: Path,
    corr_file: Path,
    cov_file: Path,
    log_file: Path,
    thread_args: List[str],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        fastspar_bin,
        "--otu_table",
        str(otu_file),
        "--correlation",
        str(corr_file),
        "--covariance",
        str(cov_file),
        *thread_args,
        "--yes",
    ]
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_fastspar_from_split_root(
    *,
    root: str | Path,
    seeds: Optional[Iterable[int]] = None,
    folds: Optional[Iterable[int]] = None,
    fastspar_bin: str = "fastspar",
    jobs: int = 1,
    threads: Optional[int] = None,
) -> List[str]:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"FastSpar root not found: {root}")
    if shutil.which(fastspar_bin) is None:
        raise FileNotFoundError(f"FastSpar executable not found in PATH: {fastspar_bin}")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")

    seed_values = (
        [int(seed) for seed in seeds]
        if seeds is not None
        else sorted(
            int(path.name.split("seed_", 1)[1])
            for path in root.glob("seed_*")
            if path.is_dir() and path.name.startswith("seed_")
        )
    )
    if not seed_values:
        raise ValueError(f"No seed directories found under {root}")

    fold_values = [int(fold) for fold in folds] if folds is not None else None
    thread_args: List[str] = []
    if threads is not None:
        if _supports_threads(fastspar_bin):
            thread_args = ["--threads", str(int(threads))]
        else:
            print(f"[WARN] {fastspar_bin} does not expose --threads; requested threads={threads} will be ignored")

    submitted = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_desc = {}
        for seed in seed_values:
            seed_dir = root / f"seed_{seed}"
            current_folds = fold_values
            if current_folds is None:
                current_folds = sorted(
                    int(path.name.split("fold_", 1)[1])
                    for path in seed_dir.glob("fold_*")
                    if path.is_dir() and path.name.startswith("fold_")
                )
            for fold in current_folds:
                fold_dir = seed_dir / f"fold_{fold}"
                input_dir = fold_dir / "fastspar" / "input"
                output_dir = fold_dir / "fastspar" / "output"
                if not fold_dir.exists():
                    print(f"[WARN] Skip missing fold directory: {fold_dir}")
                    continue
                otu_files = sorted(input_dir.glob("*_absolute_*.tsv"))
                if not otu_files:
                    raise FileNotFoundError(f"No FastSpar OTU inputs found in {input_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                for otu_file in otu_files:
                    label = otu_file.name.split("_absolute_", 1)[0]
                    corr_file = output_dir / f"{label}_median_correlation.tsv"
                    cov_file = output_dir / f"{label}_median_covariance.tsv"
                    log_file = fold_dir / "fastspar" / "logs" / f"{label}.fastspar.log"
                    desc = f"seed={seed} fold={fold} label={label}"
                    future = executor.submit(
                        _run_one,
                        fastspar_bin=fastspar_bin,
                        otu_file=otu_file,
                        corr_file=corr_file,
                        cov_file=cov_file,
                        log_file=log_file,
                        thread_args=thread_args,
                    )
                    future_to_desc[future] = desc
        for future in concurrent.futures.as_completed(future_to_desc):
            desc = future_to_desc[future]
            try:
                future.result()
                print(f"[OK] FastSpar {desc}")
                submitted.append(desc)
            except Exception as exc:
                errors.append(f"{desc}: {exc}")

    if errors:
        raise RuntimeError("One or more FastSpar tasks failed:\n" + "\n".join(errors))
    return submitted
