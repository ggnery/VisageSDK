"""Streamlit GUI for VisageSDK.

Run with:
    streamlit run gui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Side-effect imports populate registries.
import backbone  # noqa: F401
import batch_sampler  # noqa: F401
import dataset.eval  # noqa: F401
import dataset.train_val  # noqa: F401
import early_stopper  # noqa: F401
import evaluator  # noqa: F401
import loss  # noqa: F401
import transformation  # noqa: F401
from gui.run_manager import (
    RunHandle,
    find_event_files,
    launch_evaluation,
    launch_training,
    list_checkpoints,
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
    text = st.text_area(
        f"{label} — edit",
        value=load_yaml_text(selected),
        height=180,
        key=f"{key_prefix}_yaml_text",
    )
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
        bb_path, bb_text = yaml_field("Backbone", "bb", "backbone", default_pattern=backbone_name)

        st.subheader("Loss")
        loss_name = st.selectbox("Loss variant", LOSSES.names(), key="loss_name")
        loss_path, loss_text = yaml_field("Loss", "loss", "loss")

        st.subheader("Train/val dataset")
        dataset_name = st.selectbox("Dataset variant", DATASETS.names(), key="ds_name")
        ds_path, ds_text = yaml_field("Dataset", "ds", "dataset/train_val")

        st.subheader("Train/val transformation")
        tx_names = sorted(TRANSFORMATIONS.names())
        train_tx_name = st.selectbox("Train transformation", tx_names, key="train_tx")
        val_tx_name = st.selectbox("Val transformation", tx_names, key="val_tx")
        tx_path, tx_text = yaml_field("Transformation", "tx", "transformation/train_val")

    with col_right:
        st.subheader("Sampler (optional)")
        use_sampler = st.checkbox("Enable batch sampler", value=False, key="use_sampler")
        sampler_name = None
        sampler_path = None
        sampler_text = None
        if use_sampler:
            sampler_name = st.selectbox("Sampler variant", SAMPLERS.names(), key="sampler_name")
            sampler_path, sampler_text = yaml_field("Sampler", "sampler", "batch_sampler")

        st.subheader("Early stopper (optional)")
        use_es = st.checkbox("Enable early stopper", value=True, key="use_es")
        es_name = None
        es_path = None
        es_text = None
        if use_es:
            es_name = st.selectbox("Early stopper variant", EARLY_STOPPERS.names(), key="es_name")
            es_path, es_text = yaml_field("Early stopper", "es", "early_stopper")

        st.subheader("Trainer")
        trainer_path, trainer_text = yaml_field("Trainer", "trainer", "trainer")

        with st.expander("Quick overrides (modify trainer YAML)"):
            override_epochs = st.number_input("num_epochs (0 = keep YAML)", min_value=0, value=0, step=1)
            override_device = st.selectbox("device", ["", "cuda", "cpu", "mps"], index=0)
            override_seed = st.number_input("seed (-1 = keep YAML)", min_value=-1, value=-1, step=1)
            override_amp = st.selectbox("AMP", ["keep", "enable", "disable"], index=0)
            override_tb = st.selectbox("TensorBoard", ["keep", "enable", "disable"], index=0)

    st.subheader("Periodic eval (optional, written into trainer YAML)")
    use_periodic_eval = st.checkbox("Enable periodic eval", value=False, key="use_pe")
    pe_eval_dataset = pe_eval_dataset_path = None
    pe_eval_tx = pe_eval_tx_path = None
    pe_evaluator = pe_evaluator_path = None
    pe_every_n = 5
    if use_periodic_eval:
        c1, c2, c3 = st.columns(3)
        with c1:
            pe_eval_dataset = st.selectbox("Eval dataset", EVAL_DATASETS.names(), key="pe_ds")
            pe_eval_dataset_path = st.selectbox(
                "Eval dataset config",
                list_yaml_files("dataset/eval"),
                format_func=lambda p: str(p.relative_to(REPO_ROOT)),
                key="pe_ds_path",
            )
        with c2:
            pe_eval_tx = st.selectbox("Eval transformation", sorted(TRANSFORMATIONS.names()), key="pe_tx")
            pe_eval_tx_path = st.selectbox(
                "Eval transformation config",
                list_yaml_files("transformation/eval"),
                format_func=lambda p: str(p.relative_to(REPO_ROOT)),
                key="pe_tx_path",
            )
        with c3:
            pe_evaluator = st.selectbox("Evaluator", EVALUATORS.names(), key="pe_eval")
            pe_evaluator_path = st.selectbox(
                "Evaluator config",
                list_yaml_files("evaluator"),
                format_func=lambda p: str(p.relative_to(REPO_ROOT)),
                key="pe_eval_path",
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
        run_dir = make_run_dir(RUNS_PARENT, name=run_name or None)

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
        if override_epochs > 0:
            trainer_yaml["num_epochs"] = int(override_epochs)
        if override_device:
            trainer_yaml["device"] = override_device
        if override_seed >= 0:
            trainer_yaml["seed"] = int(override_seed)
        if override_amp != "keep":
            trainer_yaml.setdefault("amp", {})["enabled"] = override_amp == "enable"
        if override_tb != "keep":
            trainer_yaml.setdefault("logging", {})["tensorboard"] = override_tb == "enable"
        # Force checkpoints into the run dir
        trainer_yaml.setdefault("checkpoint", {}).setdefault("save", {})["dir"] = str(run_dir / "checkpoints")
        trainer_yaml.setdefault("logging", {}).setdefault("log_dir", str(run_dir / "tb"))
        if "tensorboard" not in trainer_yaml["logging"]:
            trainer_yaml["logging"]["tensorboard"] = True

        if use_periodic_eval:
            # All `pe_*` fields are assigned together inside the matching block above.
            assert pe_eval_dataset_path is not None
            assert pe_eval_tx_path is not None
            assert pe_evaluator_path is not None
            # Snapshot the eval YAMLs into the run dir
            pe_ds_dst = cfg_dir / "eval_dataset.yaml"
            pe_tx_dst = cfg_dir / "eval_transformation.yaml"
            pe_eval_dst = cfg_dir / "evaluator.yaml"
            write_yaml(pe_ds_dst, load_yaml_text(pe_eval_dataset_path))
            write_yaml(pe_tx_dst, load_yaml_text(pe_eval_tx_path))
            write_yaml(pe_eval_dst, load_yaml_text(pe_evaluator_path))
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
# Tab 2 — Monitor
# =============================================================================


def render_monitor_tab() -> None:
    st.header("Monitor")

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
            rc = handle.returncode
            if rc == 0:
                st.info("Status: COMPLETED")
            else:
                st.error(f"Status: EXITED ({rc})")

    # Charts
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

    if st.session_state.auto_refresh and handle and handle.is_alive:
        time.sleep(int(refresh_interval))
        st.rerun()


# =============================================================================
# Tab 3 — Evaluate
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
        bb_path, bb_text = yaml_field("Backbone", "eval_bb_yaml", "backbone", default_pattern=backbone_name)
        eval_dataset = st.selectbox("Eval dataset", EVAL_DATASETS.names(), key="eval_ds_name")
        ds_path, ds_text = yaml_field("Eval dataset", "eval_ds_yaml", "dataset/eval")
    with col2:
        eval_tx = st.selectbox("Eval transformation", sorted(TRANSFORMATIONS.names()), key="eval_tx_name")
        tx_path, tx_text = yaml_field("Eval transformation", "eval_tx_yaml", "transformation/eval")
        evaluator_name = st.selectbox("Evaluator", EVALUATORS.names(), key="evaluator_name")
        eval_path, eval_text = yaml_field("Evaluator", "evaluator_yaml", "evaluator")

    if st.button("Run evaluation", type="primary"):
        eval_run_dir = make_run_dir(RUNS_PARENT / "evals", name=selected_ckpt.stem)
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

tabs = st.tabs(["Configure & Train", "Monitor", "Evaluate"])

with tabs[0]:
    render_configure_tab()
with tabs[1]:
    render_monitor_tab()
with tabs[2]:
    render_evaluate_tab()
