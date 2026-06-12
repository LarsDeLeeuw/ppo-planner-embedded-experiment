"""
Config loader: defaults.yml → --config file → local.yml → --set CLI overrides.

Precedence is low → high; later layers replace earlier values. Mappings deep-merge;
scalars and lists replace. Validation happens once at the end via AppConfig.

The local.yml override lives next to the config package (`tracker/local.yml`),
is gitignored, and is the sanctioned place for machine-specific overrides like
camera.index, bridge.host, or SSH targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .schema import AppConfig


_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = _CONFIG_DIR / "defaults.yml"
LOCAL_OVERRIDE_PATH = _CONFIG_DIR.parent / "local.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_set_override(d: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Walk dotted_key and assign yaml.safe_load(raw_value) at the leaf."""
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError(f"--set requires a non-empty dotted key, got: {dotted_key!r}")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = yaml.safe_load(raw_value)


def build_argparser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="tracker overhead localisation and autonomous orchestrator",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="path to an experiment yaml (e.g. ../experiments/ppo_eval.yml)",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override a config field (repeatable). KEY is a dotted path, "
            "VALUE is yaml-parsed. Example: --set auto.settle.timeout_s=3.0"
        ),
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="dump the resolved config to stdout and exit",
    )
    p.add_argument(
        "--validate-config",
        action="store_true",
        help="validate and exit (no tracker launch)",
    )
    return p


def resolve_config_dict(
    config_path: Path | None,
    overrides: list[str],
) -> dict[str, Any]:
    """Build the merged config dict without instantiating AppConfig."""
    merged = _load_yaml(DEFAULTS_PATH)
    if config_path is not None:
        merged = _deep_merge(merged, _load_yaml(config_path))
    if LOCAL_OVERRIDE_PATH.exists():
        merged = _deep_merge(merged, _load_yaml(LOCAL_OVERRIDE_PATH))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set requires KEY=VALUE, got: {override!r}")
        k, v = override.split("=", 1)
        _apply_set_override(merged, k.strip(), v)
    return merged


def load_config(argv: list[str] | None = None) -> tuple[AppConfig, argparse.Namespace]:
    """Resolve config from defaults → file → local.yml → CLI overrides and validate.

    Pass `argv=[]` (or None) to load defaults + local.yml only, ignoring sys.argv.
    Pass `sys.argv[1:]` to honour the --config / --set surface.

    Returns (validated AppConfig, parsed argparse Namespace).
    """
    parser = build_argparser()
    args, _unknown = parser.parse_known_args([] if argv is None else argv)
    merged = resolve_config_dict(args.config, args.overrides)
    cfg = AppConfig.model_validate(merged)
    return cfg, args


def dump_yaml(cfg: AppConfig) -> str:
    """Serialise a resolved AppConfig to yaml."""
    raw = cfg.model_dump(mode="json")
    return yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
