"""Streamlit GUI for VisageSDK.

Run with:
    streamlit run gui/app.py
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import streamlit as st
import yaml

# `gui/` is intentionally NOT installed as a package (only src/* is — see
# pyproject.toml). When Streamlit runs the script it adds the script's own
# directory (gui/) to sys.path, so `run_manager` resolves as a flat module.
from run_manager import (  # noqa: I001
    RunHandle,
    find_event_files,
    find_eval_results_json,
    launch_evaluation,
    launch_training,
    list_checkpoints,
    list_eval_runs,
    list_existing_runs,
    make_run_dir,
    read_scalars,
    stop_run,
    tail_log,
    write_yaml,
)

from registry import (
    BACKBONES,
    DATASETS,
    EARLY_STOPPERS,
    EVAL_DATASETS,
    EVALUATORS,
    LOSSES,
    SAMPLERS,
    TRANSFORMATIONS,
)

# Trigger registry population by importing each component package for side effects.
for _component_pkg in (
    "backbone",
    "batch_sampler",
    "dataset.eval",
    "dataset.train_val",
    "early_stopper",
    "evaluator",
    "loss",
    "transformation",
):
    importlib.import_module(_component_pkg)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PARENT = REPO_ROOT / "runs"
CONFIGS_DIR = REPO_ROOT / "configs"

st.set_page_config(layout="wide", page_title="VisageSDK")
st.title("VisageSDK — Face Recognition Training")

if "training_run" not in st.session_state:
    st.session_state.training_run = None
if "eval_run" not in st.session_state:
    st.session_state.eval_run = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# =============================================================================
# Helpers
# =============================================================================


def list_yaml_files(subdir: str) -> list[Path]:
    p = CONFIGS_DIR / subdir
    if not p.exists():
        return []
    return sorted(p.rglob("*.yaml"))


def load_yaml_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def yaml_field(label: str, key_prefix: str, subdir: str, default_pattern: str = "") -> tuple[Path, str]:
    """Render a YAML picker + editor. Returns (selected_path, current_text)."""
    files = list_yaml_files(subdir)
    if not files:
        st.warning(f"No YAML files found under configs/{subdir}/")
        return Path(""), ""
    default_idx = 0
    for i, f in enumerate(files):
        if default_pattern and default_pattern in f.name:
            default_idx = i
            break
    selected = st.selectbox(
        f"{label} — config YAML",
        files,
        index=default_idx,
        format_func=lambda p: str(p.relative_to(REPO_ROOT)),
        key=f"{key_prefix}_yaml_select",
    )
    # Streamlit's text_area is sticky: once a key is in session_state, the
    # `value=` arg is ignored on later renders. Tying the widget key to the
    # selected file path gives each file its own widget identity, so switching
    # the dropdown drops a fresh widget that honors `value=` (the file's
    # disk content). Inline edits to the same file are preserved across
    # reruns because the key stays stable while the file is selected.
    file_disk_text = load_yaml_text(selected)
    text_key = f"{key_prefix}_yaml_text__{selected.name}"
    text = st.text_area(
        f"{label} — edit",
        value=file_disk_text,
        height=180,
        key=text_key,
    )
    # Final safety net: if for any reason the editor still came back empty
    # while the file on disk has content, fall back to the disk text. Prevents
    # silent zero-byte snapshots that crash eval.py / train.py downstream.
    if not text.strip() and file_disk_text.strip():
        text = file_disk_text
    return selected, text


# =============================================================================
# Tab 1 — Configure & Train
# =============================================================================


def render_configure_tab() -> None:
    st.header("Configure & Train")

    if st.session_state.training_run and st.session_state.training_run.is_alive:
        st.success(f"Training is RUNNING in: {st.session_state.training_run.run_dir}")
    elif st.session_state.training_run is not None:
        rc = st.session_state.training_run.returncode
        if rc == 0:
            st.info(f"Last run completed successfully: {st.session_state.training_run.run_dir}")
        else:
            st.error(f"Last run exited with code {rc}: {st.session_state.training_run.run_dir}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Backbone")
        backbone_name = st.selectbox("Backbone variant", BACKBONES.names(), key="bb_name")
        _, bb_text = yaml_field("Backbone", "bb", "backbone", default_pattern=backbone_name)

        st.subheader("Loss")
        loss_name = st.selectbox("Loss variant", LOSSES.names(), key="loss_name")
        _, loss_text = yaml_field("Loss", "loss", "loss")

        st.subheader("Train/val dataset")
        dataset_name = st.selectbox("Dataset variant", DATASETS.names(), key="ds_name")
        _, ds_text = yaml_field("Dataset", "ds", "dataset/train_val")

        st.subheader("Train/val transformation")
        # Filter by naming convention so the user can't accidentally pick
        # `lfw_eval` (which expects `{normalize: {...}}`) for the train
        # split (which expects `{train: {...}, val: {...}}`) — that
        # mismatch crashes inside the transformation with a confusing
        # AttributeError. Same goes for picking a `_val` transform as
        # the train transformation.
        tx_names = sorted(TRANSFORMATIONS.names())
        train_tx_candidates = [n for n in tx_names if n.endswith("_train")] or tx_names
        val_tx_candidates = [n for n in tx_names if n.endswith("_val")] or tx_names
        train_tx_name = st.selectbox("Train transformation", train_tx_candidates, key="train_tx")
        val_tx_name = st.selectbox("Val transformation", val_tx_candidates, key="val_tx")
        _, tx_text = yaml_field("Transformation", "tx", "transformation/train_val")

    with col_right:
        st.subheader("Sampler (optional)")
        use_sampler = st.checkbox("Enable batch sampler", value=False, key="use_sampler")
        sampler_name = None
        sampler_text = None
        if use_sampler:
            sampler_name = st.selectbox("Sampler variant", SAMPLERS.names(), key="sampler_name")
            _, sampler_text = yaml_field("Sampler", "sampler", "batch_sampler")

        st.subheader("Early stopper (optional)")
        use_es = st.checkbox("Enable early stopper", value=True, key="use_es")
        es_name = None
        es_text = None
        if use_es:
            es_name = st.selectbox("Early stopper variant", EARLY_STOPPERS.names(), key="es_name")
            _, es_text = yaml_field("Early stopper", "es", "early_stopper")

        st.subheader("Trainer")
        _, trainer_text = yaml_field("Trainer", "trainer", "trainer")

        with st.expander("Quick overrides (modify trainer YAML)"):
            override_epochs = st.number_input("num_epochs (0 = keep YAML)", min_value=0, value=0, step=1)
            override_device = st.selectbox("device", ["", "cuda", "cpu", "mps"], index=0)
            override_seed = st.number_input("seed (-1 = keep YAML)", min_value=-1, value=-1, step=1)
            override_amp = st.selectbox("AMP", ["keep", "enable", "disable"], index=0)
            override_tb = st.selectbox("TensorBoard", ["keep", "enable", "disable"], index=0)

    st.subheader("Periodic eval (optional, written into trainer YAML)")
    use_periodic_eval = st.checkbox("Enable periodic eval", value=False, key="use_pe")
    pe_eval_dataset = pe_eval_dataset_path = pe_ds_text = None
    pe_eval_tx = pe_eval_tx_path = pe_tx_text = None
    pe_evaluator = pe_evaluator_path = pe_eval_text = None
    pe_every_n = 5
    if use_periodic_eval:
        c1, c2, c3 = st.columns(3)
        with c1:
            pe_eval_dataset = st.selectbox("Eval dataset", EVAL_DATASETS.names(), key="pe_ds")
            pe_eval_dataset_path, pe_ds_text = yaml_field(
                "Eval dataset", "pe_ds_yaml", "dataset/eval"
            )
        with c2:
            eval_tx_names = (
                [n for n in sorted(TRANSFORMATIONS.names()) if n.endswith("_eval")]
                or sorted(TRANSFORMATIONS.names())
            )
            pe_eval_tx = st.selectbox("Eval transformation", eval_tx_names, key="pe_tx")
            pe_eval_tx_path, pe_tx_text = yaml_field(
                "Eval transformation", "pe_tx_yaml", "transformation/eval"
            )
        with c3:
            pe_evaluator = st.selectbox("Evaluator", EVALUATORS.names(), key="pe_eval")
            pe_evaluator_path, pe_eval_text = yaml_field(
                "Evaluator", "pe_eval_yaml", "evaluator"
            )
        pe_every_n = st.number_input("Run every N epochs", min_value=1, value=5, step=1)

    st.divider()
    run_name = st.text_input("Run name (optional)", value="", help="Appended to the timestamped run dir")
    launch_btn = st.button(
        "Launch Training",
        type="primary",
        disabled=bool(st.session_state.training_run and st.session_state.training_run.is_alive),
    )

    if launch_btn:
        # Refuse to launch if any required YAML editor came back empty —
        # this used to silently produce zero-byte snapshots and crash
        # train.py with confusing AttributeErrors deep in the config loader.
        # Mirrors the same guard already in render_evaluate_tab.
        empty_required = [
            label
            for label, content in (
                ("backbone", bb_text),
                ("loss", loss_text),
                ("dataset", ds_text),
                ("transformation", tx_text),
                ("trainer", trainer_text),
            )
            if not content.strip()
        ]
        if use_sampler and sampler_text is not None and not sampler_text.strip():
            empty_required.append("sampler")
        if use_es and es_text is not None and not es_text.strip():
            empty_required.append("early_stopper")
        if use_periodic_eval:
            for label, content in (
                ("periodic eval_dataset", pe_ds_text),
                ("periodic eval_transformation", pe_tx_text),
                ("periodic evaluator", pe_eval_text),
            ):
                if content is not None and not content.strip():
                    empty_required.append(label)
        if empty_required:
            st.error(
                "Empty YAML editor(s): "
                + ", ".join(empty_required)
                + ". Restart Streamlit (Ctrl+C and re-run) so the editors reload from disk."
            )
            return

        run_dir = make_run_dir(RUNS_PARENT / "trains", name=run_name or None)

        # Write YAMLs out (with overrides applied to trainer)
        cfg_dir = run_dir / "configs"
        write_yaml(cfg_dir / "backbone.yaml", bb_text)
        write_yaml(cfg_dir / "loss.yaml", loss_text)
        write_yaml(cfg_dir / "dataset.yaml", ds_text)
        write_yaml(cfg_dir / "transformation.yaml", tx_text)

        if use_sampler:
            assert sampler_text is not None
            write_yaml(cfg_dir / "sampler.yaml", sampler_text)
        if use_es:
            assert es_text is not None
            write_yaml(cfg_dir / "early_stopper.yaml", es_text)

        trainer_yaml = yaml.safe_load(trainer_text) or {}

        def _ensure_dict(d: dict, key: str) -> dict:
            """`setdefault(key, {})` returns the existing value, even if it's None
            (e.g. user wrote `checkpoint: null`). This guarantees a dict back."""
            if d.get(key) is None:
                d[key] = {}
            return d[key]

        if override_epochs > 0:
            trainer_yaml["num_epochs"] = int(override_epochs)
        if override_device:
            trainer_yaml["device"] = override_device
        if override_seed >= 0:
            trainer_yaml["seed"] = int(override_seed)
        if override_amp != "keep":
            _ensure_dict(trainer_yaml, "amp")["enabled"] = override_amp == "enable"
        if override_tb != "keep":
            _ensure_dict(trainer_yaml, "logging")["tensorboard"] = override_tb == "enable"
        # Force checkpoints into the run dir
        ckpt = _ensure_dict(trainer_yaml, "checkpoint")
        _ensure_dict(ckpt, "save")["dir"] = str(run_dir / "checkpoints")
        logging_block = _ensure_dict(trainer_yaml, "logging")
        logging_block.setdefault("log_dir", str(run_dir / "tb"))
        if "tensorboard" not in logging_block:
            logging_block["tensorboard"] = True

        if use_periodic_eval:
            # All `pe_*` fields are assigned together inside the matching block above.
            assert pe_eval_dataset_path is not None
            assert pe_eval_tx_path is not None
            assert pe_evaluator_path is not None
            assert pe_ds_text is not None
            assert pe_tx_text is not None
            assert pe_eval_text is not None
            # Snapshot the (possibly inline-edited) eval YAMLs into the run dir.
            pe_ds_dst = cfg_dir / "eval_dataset.yaml"
            pe_tx_dst = cfg_dir / "eval_transformation.yaml"
            pe_eval_dst = cfg_dir / "evaluator.yaml"
            write_yaml(pe_ds_dst, pe_ds_text)
            write_yaml(pe_tx_dst, pe_tx_text)
            write_yaml(pe_eval_dst, pe_eval_text)
            trainer_yaml["periodic_eval"] = {
                "enabled": True,
                "every_n_epochs": int(pe_every_n),
                "dataset": pe_eval_dataset,
                "dataset_config": str(pe_ds_dst),
                "transformation": pe_eval_tx,
                "transformation_config": str(pe_tx_dst),
                "evaluator": pe_evaluator,
                "evaluator_config": str(pe_eval_dst),
            }

        write_yaml(cfg_dir / "trainer.yaml", trainer_yaml)

        env: dict[str, str] = {
            "BACKBONE": backbone_name,
            "BACKBONE_CONFIG": str(cfg_dir / "backbone.yaml"),
            "LOSS": loss_name,
            "LOSS_CONFIG": str(cfg_dir / "loss.yaml"),
            "TRAIN_VAL_DATASET": dataset_name,
            "TRAIN_VAL_DATASET_CONFIG": str(cfg_dir / "dataset.yaml"),
            "TRAIN_TRANSFORMATION": train_tx_name,
            "VAL_TRANSFORMATION": val_tx_name,
            "TRAIN_VAL_TRANSFORMATION_CONFIG": str(cfg_dir / "transformation.yaml"),
            "TRAINER_CONFIG": str(cfg_dir / "trainer.yaml"),
        }
        if use_sampler:
            assert sampler_name is not None
            env["SAMPLER"] = sampler_name
            env["SAMPLER_CONFIG"] = str(cfg_dir / "sampler.yaml")
        if use_es:
            assert es_name is not None
            env["EARLY_STOPPER"] = es_name
            env["EARLY_STOPPER_CONFIG"] = str(cfg_dir / "early_stopper.yaml")

        handle = launch_training(env, run_dir)
        st.session_state.training_run = handle
        st.success(f"Training launched in {run_dir}. Switch to the Monitor tab.")
        time.sleep(1)
        st.rerun()


# =============================================================================
# Tab 2 — Monitor Train
# =============================================================================


def render_monitor_tab() -> None:
    st.header("Monitor Train")

    runs = list_existing_runs(RUNS_PARENT)
    if not runs:
        st.info("No runs yet. Launch one from the Configure & Train tab.")
        return

    run_options = [str(r.relative_to(REPO_ROOT)) for r in runs]
    if st.session_state.training_run:
        run_options = ["(current)"] + run_options
    selection = st.selectbox("Run", run_options, key="monitor_run")
    run_dir = st.session_state.training_run.run_dir if selection == "(current)" else REPO_ROOT / selection

    handle: RunHandle | None = (
        st.session_state.training_run
        if st.session_state.training_run and st.session_state.training_run.run_dir == run_dir
        else None
    )

    cols = st.columns([1, 1, 1, 1, 4])
    with cols[0]:
        st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    with cols[1]:
        refresh_interval = st.number_input("Interval (s)", min_value=2, max_value=60, value=5)
    with cols[2]:
        if st.button("Refresh now"):
            st.rerun()
    with cols[3]:
        if handle and handle.is_alive and st.button("Stop run", type="secondary"):
            stop_run(handle)
            st.rerun()

    if handle:
        if handle.is_alive and handle.process is not None:
            st.success(f"Status: RUNNING — pid {handle.process.pid}")
        else:
            # Subprocess finished; close its log handle now (idempotent).
            handle.cleanup()
            rc = handle.returncode
            if rc == 0:
                st.info("Status: COMPLETED")
            else:
                st.error(f"Status: EXITED ({rc})")

    # Charts and log are wrapped in a fragment so auto-refresh re-renders ONLY
    # them (without freezing the controls above with `time.sleep`). When
    # `auto_refresh` is off or the run is finished, run_every=None disables
    # the periodic re-run.
    auto_run_every = (
        int(refresh_interval) if (st.session_state.auto_refresh and handle and handle.is_alive) else None
    )

    @st.fragment(run_every=auto_run_every)
    def _live_section() -> None:
        event_files = find_event_files(run_dir)
        scalars = read_scalars(event_files)

        if not scalars:
            st.info("No TensorBoard events yet. Charts will appear once the first epoch logs.")
        else:
            loss_tags = sorted(t for t in scalars if t.startswith("loss/"))
            eval_tags = sorted(t for t in scalars if t.startswith("eval/"))
            train_stat_tags = sorted(t for t in scalars if t.startswith("train_stats/"))
            val_stat_tags = sorted(t for t in scalars if t.startswith("val_stats/"))
            lr_tags = sorted(t for t in scalars if t == "lr")

            def plot_group(title: str, tags: list[str]) -> None:
                if not tags:
                    return
                st.subheader(title)
                data = {tag: dict(scalars[tag]) for tag in tags}
                steps = sorted({s for series in data.values() for s in series})
                chart_data: dict[str, list] = {tag: [data[tag].get(s, None) for s in steps] for tag in tags}
                chart_data["epoch"] = list(steps)
                import pandas as pd

                df = pd.DataFrame(chart_data).set_index("epoch")
                st.line_chart(df)

            plot_group("Loss", loss_tags)
            plot_group("Learning rate", lr_tags)
            plot_group("Eval metrics", eval_tags)
            plot_group("Training stats", train_stat_tags)
            plot_group("Validation stats", val_stat_tags)

        st.divider()
        st.subheader("Log tail")
        log_path = run_dir / "train.log"
        if not log_path.exists():
            log_path = next(iter(run_dir.glob("*.log")), None)
        st.code(tail_log(log_path, max_lines=200) or "(no log yet)", language="text")

    _live_section()


# =============================================================================
# Tab 3 — Monitor Eval
# =============================================================================


def render_monitor_eval_tab() -> None:
    st.header("Monitor Eval")

    runs = list_eval_runs(RUNS_PARENT)
    if not runs:
        st.info("No eval runs yet. Launch one from the Evaluate tab.")
        return

    run_options = [str(r.relative_to(REPO_ROOT)) for r in runs]
    # Pin the in-flight run at the top so the user lands on it after launching.
    if st.session_state.eval_run:
        run_options = ["(current)"] + run_options
    selection = st.selectbox("Eval run", run_options, key="monitor_eval_run")
    run_dir = (
        st.session_state.eval_run.run_dir if selection == "(current)" else REPO_ROOT / selection
    )

    handle: RunHandle | None = (
        st.session_state.eval_run
        if st.session_state.eval_run and st.session_state.eval_run.run_dir == run_dir
        else None
    )

    cols = st.columns([1, 1, 1, 1, 4])
    with cols[0]:
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh", value=st.session_state.auto_refresh, key="monitor_eval_auto"
        )
    with cols[1]:
        refresh_interval = st.number_input(
            "Interval (s)", min_value=2, max_value=60, value=5, key="monitor_eval_interval"
        )
    with cols[2]:
        if st.button("Refresh now", key="monitor_eval_refresh"):
            st.rerun()
    with cols[3]:
        if handle and handle.is_alive and st.button("Stop run", type="secondary", key="monitor_eval_stop"):
            stop_run(handle)
            st.rerun()

    if handle:
        if handle.is_alive and handle.process is not None:
            st.success(f"Status: RUNNING — pid {handle.process.pid}")
        else:
            handle.cleanup()
            rc = handle.returncode
            if rc == 0:
                st.info("Status: COMPLETED")
            else:
                st.error(f"Status: EXITED ({rc})")

    # eval.py writes its results JSON next to the original checkpoint, not
    # inside the eval run dir. find_eval_results_json reads env.json from the
    # eval run dir to recover the checkpoint path and locate the file.
    auto_run_every = (
        int(refresh_interval) if (st.session_state.auto_refresh and handle and handle.is_alive) else None
    )

    @st.fragment(run_every=auto_run_every)
    def _live_section() -> None:
        results_json = find_eval_results_json(run_dir)
        if results_json is not None and results_json.exists():
            import json as jsonlib

            with open(results_json) as f:
                data = jsonlib.load(f)
            results = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(results, dict) and results:
                _render_eval_results(results)
            else:
                st.json(data)
            st.caption(f"Source: {results_json.relative_to(REPO_ROOT)}")
        else:
            st.info("Results JSON not found yet — eval might still be running or have crashed.")

        st.divider()
        st.subheader("Log tail")
        log_path = run_dir / "eval.log"
        if not log_path.exists():
            log_path = next(iter(run_dir.glob("*.log")), None)
        st.code(tail_log(log_path, max_lines=200) or "(no log yet)", language="text")

    _live_section()


def _classify_eval_metric(name: str) -> str:
    """Bucket an eval metric for display: section + value formatting."""
    n = name.lower()
    if any(t in n for t in ("rank_", "cmc")):
        return "identification_rank"
    if n == "map":
        return "identification_map"
    if "tar@far" in n:
        return "tar"
    if "threshold@far" in n:
        return "tar_threshold"
    if "threshold" in n:
        return "threshold"
    if "auc" in n:
        return "auc"
    if "eer" in n:
        return "eer"
    if "accuracy" in n:
        return "accuracy"
    return "other"


def _format_eval_metric(name: str, value: float) -> str:
    bucket = _classify_eval_metric(name)
    # Treat anything in [0, 1] that conceptually represents a rate / accuracy
    # as a percentage. AUC and raw threshold values stay as decimals.
    if bucket in {"identification_rank", "identification_map", "tar", "eer", "accuracy"}:
        return f"{value * 100:.2f}%"
    if bucket in {"auc", "threshold", "tar_threshold"}:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _render_eval_results(results: dict) -> None:
    """Pretty-print the eval JSON: headline cards + per-section tables.

    Auto-detects whether we're looking at identification or verification
    metrics (or both), so the same renderer works for either evaluator."""
    import pandas as pd

    scalars: dict[str, float] = {
        k: float(v) for k, v in results.items() if isinstance(v, (int, float))
    }
    if not scalars:
        st.json(results)
        return

    # ── Headline metrics row ────────────────────────────────────────────
    st.subheader("Headline")
    headline_keys = [
        ("rank_1", "Rank-1"),
        ("rank_5", "Rank-5"),
        ("mAP", "mAP"),
        ("lfw_accuracy_mean", "LFW accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("eer", "EER"),
        ("tar@far=1e-03", "TAR@FAR=1e-3"),
    ]
    available = [(key, label) for key, label in headline_keys if key in scalars]
    if available:
        cols = st.columns(len(available))
        for col, (key, label) in zip(cols, available, strict=False):
            value = scalars[key]
            display = _format_eval_metric(key, value)
            # Tack the std onto the LFW accuracy headline so the dispersion is
            # visible without paging through the detailed table below.
            if key == "lfw_accuracy_mean":
                std = scalars.get("lfw_accuracy_std")
                if std is not None:
                    display = f"{value * 100:.2f}% ± {std * 100:.2f}%"
            col.metric(label, display)

    # ── Sectioned breakdown ────────────────────────────────────────────
    section_titles: dict[str, str] = {
        "identification_rank": "Identification — Rank / CMC",
        "identification_map": "Identification — mAP",
        "accuracy": "Verification — Accuracy",
        "auc": "Verification — ROC-AUC",
        "eer": "Verification — EER",
        "tar": "TAR @ FAR target",
        "tar_threshold": "Threshold @ FAR target",
        "threshold": "Other thresholds",
        "other": "Other",
    }
    grouped: dict[str, list[tuple[str, str, float]]] = {}
    for k in sorted(scalars):
        bucket = _classify_eval_metric(k)
        grouped.setdefault(bucket, []).append((k, _format_eval_metric(k, scalars[k]), scalars[k]))

    for bucket, title in section_titles.items():
        rows = grouped.get(bucket)
        if not rows:
            continue
        st.subheader(title)
        df = pd.DataFrame(rows, columns=["metric", "value", "raw"]).set_index("metric")
        st.dataframe(df, use_container_width=True)

    # ── Curves (ROC etc.) ───────────────────────────────────────────────
    roc = results.get("roc_curve")
    if isinstance(roc, dict) and "fpr" in roc and "tpr" in roc:
        st.subheader("ROC curve")
        _plot_roc_curve(roc["fpr"], roc["tpr"], auc=scalars.get("roc_auc"), eer=scalars.get("eer"))

    score_dists = results.get("score_distributions")
    if isinstance(score_dists, dict) and "genuine" in score_dists and "impostor" in score_dists:
        st.subheader("Genuine vs impostor score distributions")
        # Pick the most informative threshold lines available in scalars: EER
        # (operating point of the ROC) and TAR@FAR=1e-3 (typical biometric
        # acceptance threshold). Both are in distance space.
        threshold_lines = []
        if "eer_threshold" in scalars:
            threshold_lines.append(("EER", float(scalars["eer_threshold"])))
        if "threshold@far=1e-03" in scalars:
            threshold_lines.append(("FAR=1e-3", float(scalars["threshold@far=1e-03"])))
        _plot_score_distributions(
            genuine=score_dists["genuine"],
            impostor=score_dists["impostor"],
            distance_kind=str(score_dists.get("distance_kind", "cosine")),
            threshold_lines=threshold_lines,
        )

    # ── Anything else (rare) ────────────────────────────────────────────
    leftover = {
        k: v
        for k, v in results.items()
        if not isinstance(v, (int, float)) and k not in {"roc_curve", "score_distributions"}
    }
    if leftover:
        st.subheader("Other (non-scalar)")
        st.json(leftover)


def _plot_roc_curve(
    fpr: list[float], tpr: list[float], auc: float | None = None, eer: float | None = None
) -> None:
    """Render a verification ROC curve in linear and log-x views side by side.

    Log-x highlights the low-FAR operating regime that matters for
    biometric thresholds; the linear view is for at-a-glance shape and
    AUC. The two share data so they always agree."""
    import numpy as np
    import matplotlib.pyplot as plt

    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]

    auc_label = f"AUC = {auc:.3f}" if auc is not None else "ROC"
    title_extra = f" (EER = {eer * 100:.2f}%)" if eer is not None else ""

    col_linear, col_log = st.columns(2)
    with col_linear:
        fig_lin, ax_lin = plt.subplots(figsize=(4.5, 4.5))
        ax_lin.plot(fpr_arr, tpr_arr, color="C0", label=auc_label, linewidth=2)
        ax_lin.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, label="Chance")
        ax_lin.set_xlabel("False Positive Rate")
        ax_lin.set_ylabel("True Positive Rate")
        ax_lin.set_xlim(0, 1)
        ax_lin.set_ylim(0, 1.02)
        ax_lin.set_title(f"Linear{title_extra}")
        ax_lin.legend(loc="lower right")
        ax_lin.grid(True, alpha=0.3)
        st.pyplot(fig_lin)
        plt.close(fig_lin)

    with col_log:
        fig_log, ax_log = plt.subplots(figsize=(4.5, 4.5))
        # Replace zero FPR with a tiny positive so the log scale doesn't drop
        # those points off the left edge.
        floor = max(1e-6, float(fpr_arr[fpr_arr > 0].min()) if (fpr_arr > 0).any() else 1e-3)
        x = np.where(fpr_arr <= 0, floor, fpr_arr)
        ax_log.plot(x, tpr_arr, color="C0", label=auc_label, linewidth=2)
        ax_log.set_xscale("log")
        ax_log.set_xlim(floor, 1)
        ax_log.set_ylim(0, 1.02)
        ax_log.set_xlabel("False Positive Rate (log)")
        ax_log.set_ylabel("True Positive Rate")
        ax_log.set_title("Log-FPR — operating regime")
        ax_log.legend(loc="lower right")
        ax_log.grid(True, which="both", alpha=0.3)
        st.pyplot(fig_log)
        plt.close(fig_log)


def _plot_score_distributions(
    genuine: list[float],
    impostor: list[float],
    distance_kind: str = "cosine",
    threshold_lines: list[tuple[str, float]] | None = None,
) -> None:
    """Overlapping histograms of genuine vs impostor pair distances.

    The visual gap between the two distributions is the actual signal a
    threshold-based verifier exploits. When clusters are loose (cross-
    entropy + small dataset), the two histograms bleed into each other;
    a margin-based loss should push genuine left and impostor right."""
    import matplotlib.pyplot as plt
    import numpy as np

    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    if g.size == 0 or i.size == 0:
        st.info("Not enough pairs to plot the score distributions.")
        return

    # Shared bin edges so the two histograms are directly comparable; bin
    # count scales with sqrt(n) but capped to keep the plot readable.
    n_bins = int(min(60, max(20, np.sqrt(max(g.size, i.size)))))
    lo = float(min(g.min(), i.min()))
    hi = float(max(g.max(), i.max()))
    # matplotlib accepts Sequence[float] (not numpy array directly per its
    # stubs), so build the edges as a Python list.
    bins: list[float] = np.linspace(lo, hi, n_bins + 1).tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(g, bins=bins, alpha=0.55, color="#2ca02c", label=f"Genuine (n={g.size})")
    ax.hist(i, bins=bins, alpha=0.55, color="#d62728", label=f"Impostor (n={i.size})")

    if threshold_lines:
        # Stagger the line styles so multiple thresholds remain distinguishable
        # when they fall close together (typical for tight ROC operating points).
        styles = [("--", "C0"), (":", "C4"), ("-.", "C5")]
        for (label, value), (linestyle, color) in zip(threshold_lines, styles, strict=False):
            ax.axvline(value, linestyle=linestyle, color=color, linewidth=1.5, label=f"{label} thr = {value:.3f}")

    ax.set_xlabel(f"{distance_kind.capitalize()} distance — lower = more similar")
    ax.set_ylabel("Pair count")
    ax.set_title("Score distributions: genuine vs impostor")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Genuine: median {np.median(g):.3f}, IQR [{np.percentile(g, 25):.3f}, {np.percentile(g, 75):.3f}]"
        f"  ·  Impostor: median {np.median(i):.3f}, IQR [{np.percentile(i, 25):.3f}, {np.percentile(i, 75):.3f}]"
    )


# =============================================================================
# Tab 4 — Evaluate
# =============================================================================


def render_evaluate_tab() -> None:
    st.header("Evaluate")

    if st.session_state.eval_run and st.session_state.eval_run.is_alive:
        st.success(f"Eval RUNNING in: {st.session_state.eval_run.run_dir}")
    elif st.session_state.eval_run is not None:
        rc = st.session_state.eval_run.returncode
        if rc == 0:
            st.info("Last eval completed.")
        else:
            st.error(f"Last eval exited with code {rc}")

    runs = list_existing_runs(RUNS_PARENT)
    if not runs:
        st.info("No runs found.")
        return

    selected_run = st.selectbox(
        "Run with checkpoint",
        runs,
        format_func=lambda r: str(r.relative_to(REPO_ROOT)),
        key="eval_run_select",
    )
    checkpoints = list_checkpoints(selected_run)
    if not checkpoints:
        st.warning("No checkpoints in this run.")
        return
    selected_ckpt = st.selectbox(
        "Checkpoint",
        checkpoints,
        format_func=lambda p: str(p.relative_to(REPO_ROOT)),
        key="eval_ckpt",
    )

    col1, col2 = st.columns(2)
    with col1:
        backbone_name = st.selectbox("Backbone variant", BACKBONES.names(), key="eval_bb")
        _, bb_text = yaml_field("Backbone", "eval_bb_yaml", "backbone", default_pattern=backbone_name)
        eval_dataset = st.selectbox("Eval dataset", EVAL_DATASETS.names(), key="eval_ds_name")
        _, ds_text = yaml_field("Eval dataset", "eval_ds_yaml", "dataset/eval")
    with col2:
        eval_tx_names = (
            [n for n in sorted(TRANSFORMATIONS.names()) if n.endswith("_eval")]
            or sorted(TRANSFORMATIONS.names())
        )
        eval_tx = st.selectbox("Eval transformation", eval_tx_names, key="eval_tx_name")
        _, tx_text = yaml_field("Eval transformation", "eval_tx_yaml", "transformation/eval")
        evaluator_name = st.selectbox("Evaluator", EVALUATORS.names(), key="evaluator_name")
        _, eval_text = yaml_field("Evaluator", "evaluator_yaml", "evaluator")

    st.divider()
    eval_run_name = st.text_input(
        "Run name (optional)",
        value="",
        help="Appended to the timestamped eval run dir under runs/evals/.",
        key="eval_run_name",
    )

    if st.button("Run evaluation", type="primary"):
        # Refuse to launch if any YAML editor came back empty — happens when the
        # text_area's session_state was never populated and would otherwise yield
        # zero-byte YAMLs that crash eval.py downstream with confusing errors
        # like 'BackboneConfig has no attribute input_size'.
        empty = [
            label
            for label, content in (
                ("backbone", bb_text),
                ("eval_dataset", ds_text),
                ("eval_transformation", tx_text),
                ("evaluator", eval_text),
            )
            if not content.strip()
        ]
        if empty:
            st.error(
                "Empty YAML editor(s): "
                + ", ".join(empty)
                + ". Restart Streamlit (Ctrl+C and re-run the command) so the "
                "editors reload from disk."
            )
            return

        suffix = eval_run_name or selected_ckpt.stem
        eval_run_dir = make_run_dir(RUNS_PARENT / "evals", name=suffix)
        cfg_dir = eval_run_dir / "configs"
        write_yaml(cfg_dir / "backbone.yaml", bb_text)
        write_yaml(cfg_dir / "eval_dataset.yaml", ds_text)
        write_yaml(cfg_dir / "eval_transformation.yaml", tx_text)
        write_yaml(cfg_dir / "evaluator.yaml", eval_text)

        env = {
            "BACKBONE": backbone_name,
            "BACKBONE_CONFIG": str(cfg_dir / "backbone.yaml"),
            "CHECKPOINT_PATH": str(selected_ckpt),
            "EVAL_DATASET": eval_dataset,
            "EVAL_DATASET_CONFIG": str(cfg_dir / "eval_dataset.yaml"),
            "EVAL_TRANSFORMATION": eval_tx,
            "EVAL_TRANSFORMATION_CONFIG": str(cfg_dir / "eval_transformation.yaml"),
            "EVALUATOR": evaluator_name,
            "EVALUATOR_CONFIG": str(cfg_dir / "evaluator.yaml"),
        }
        st.session_state.eval_run = launch_evaluation(env, eval_run_dir)
        st.rerun()

    if st.session_state.eval_run is not None:
        st.subheader("Eval log")
        st.code(
            tail_log(st.session_state.eval_run.log_path, max_lines=200) or "(no output yet)",
            language="text",
        )

        # Show the JSON results next to the original checkpoint, if eval.py wrote one
        ckpt_dir = selected_ckpt.parent
        eval_jsons = sorted(ckpt_dir.glob("eval_*.json"), reverse=True)
        if eval_jsons and not st.session_state.eval_run.is_alive:
            st.subheader("Latest eval results")
            import json as jsonlib

            with open(eval_jsons[0]) as f:
                data = jsonlib.load(f)
            st.json(data)


# =============================================================================
# Layout
# =============================================================================

tabs = st.tabs(["Configure & Train", "Monitor Train", "Monitor Eval", "Evaluate"])

with tabs[0]:
    render_configure_tab()
with tabs[1]:
    render_monitor_tab()
with tabs[2]:
    render_monitor_eval_tab()
with tabs[3]:
    render_evaluate_tab()
