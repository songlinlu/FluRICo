from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .lgbm_search_space import lgbm_grid_by_profile


DEFAULT_SEEDS = [2026, 3407, 5279, 6841, 9187]


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        token = str(value)
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def parse_label_map_entries(entries: Optional[Iterable[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries or []:
        token = str(entry).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid --label-map item '{token}'. Expected format: source:target")
        source, target = token.split(":", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid --label-map item '{token}'. Empty source or target.")
        mapping[source] = target
    return mapping


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    labels: List[str]
    label_map: Dict[str, str] = field(default_factory=dict)
    positive_label: Optional[str] = None

    def final_labels(self) -> List[str]:
        mapped = [self.label_map.get(label, label) for label in self.labels]
        return ordered_unique(mapped)

    @property
    def is_binary(self) -> bool:
        return len(self.final_labels()) == 2

    @property
    def n_classes(self) -> int:
        return len(self.final_labels())

    def validate(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name cannot be empty")
        if not self.labels:
            raise ValueError("labels cannot be empty")
        final_labels = self.final_labels()
        if len(final_labels) < 2:
            raise ValueError(
                "TaskSpec must contain at least two final classes after applying label_map"
            )
        if self.positive_label is not None and self.positive_label not in final_labels:
            raise ValueError(
                f"positive_label='{self.positive_label}' not found in final labels: {final_labels}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "labels": list(self.labels),
            "label_map": dict(self.label_map),
            "positive_label": self.positive_label,
            "final_labels": self.final_labels(),
        }


@dataclass
class RunConfig:
    out_dir: Path
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    n_splits: int = 5
    zero_max_fraction: Optional[float] = 0.8
    n_jobs: int = 1
    fastspar_bin: str = "fastspar"
    fastspar_jobs: int = 1
    fastspar_threads: Optional[int] = None
    flurico_use_fast: bool = False
    flurico_n_jobs: int = 1
    mb_top_n: int = 50
    mt_top_n: int = 20
    lgbm_profile: str = "lgbm_grid"
    tie_eps: float = 1e-6
    full_prefix: str = "modeling_all_features_lgbm"
    selected_prefix: str = "modeling_selected_features_lgbm"
    flurico_log_microbe: bool = True
    flurico_log_metabolite: bool = True
    model_log_microbe: bool = True
    model_log_metabolite: bool = True

    def validate(self) -> None:
        self.out_dir = Path(self.out_dir).resolve()
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        if self.zero_max_fraction is not None and not (0.0 < self.zero_max_fraction <= 1.0):
            raise ValueError("zero_max_fraction must be in (0, 1]")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.fastspar_jobs < 1:
            raise ValueError("fastspar_jobs must be >= 1")
        if self.fastspar_threads is not None and self.fastspar_threads < 1:
            raise ValueError("fastspar_threads must be >= 1")
        if self.flurico_n_jobs < 1:
            raise ValueError("flurico_n_jobs must be >= 1")
        if self.mb_top_n < 1 or self.mt_top_n < 1:
            raise ValueError("mb_top_n and mt_top_n must be >= 1")
        # Validate early to surface unsupported profile names before long runs.
        lgbm_grid_by_profile(self.lgbm_profile)
