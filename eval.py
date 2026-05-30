import json
from datetime import datetime
from pathlib import Path

from config import ENVEvalConfig
from tools.evaluator_builder import EvaluatorBuilder


def main(env: ENVEvalConfig) -> None:
    builder = EvaluatorBuilder(env)
    evaluator = builder.build()

    print(builder.config_str)
    print("Running evaluation...")
    results = evaluator.evaluate()

    print()
    print("=" * 50)
    print(f"{type(evaluator).__name__} RESULTS")
    print("=" * 50)
    scalar_items = [
        (k, v) for k, v in results.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    width = max((len(k) for k, _ in scalar_items), default=0)
    for k, v in scalar_items:
        print(f"{k:<{width}} : {v:.6f}")
    # Note any non-scalar payloads (e.g. roc_curve points) so it's obvious
    # they're persisted in the JSON even though they don't fit the table.
    non_scalar_keys = [
        k for k, v in results.items() if not isinstance(v, (int, float)) or isinstance(v, bool)
    ]
    if non_scalar_keys:
        print("-" * 50)
        for k in non_scalar_keys:
            print(f"{k:<{width}} : <non-scalar, see JSON>")
    print("=" * 50)

    ckpt_dir = Path(env.checkpoint_path).parent
    out = ckpt_dir / f"eval_{type(evaluator).__name__}_{datetime.now().isoformat(timespec='seconds')}.json"
    with open(out, "w") as f:
        json.dump({"checkpoint": env.checkpoint_path, "results": results}, f, indent=2)
    print(f"Saved results to: {out}")


if __name__ == "__main__":
    env = ENVEvalConfig.from_env()
    main(env)
