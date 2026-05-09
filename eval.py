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
    width = max((len(k) for k in results), default=0)
    for k, v in results.items():
        print(f"{k:<{width}} : {v:.6f}")
    print("=" * 50)

    ckpt_dir = Path(env.checkpoint_path).parent
    out = ckpt_dir / f"eval_{type(evaluator).__name__}_{datetime.now().isoformat(timespec='seconds')}.json"
    with open(out, "w") as f:
        json.dump({"checkpoint": env.checkpoint_path, "results": results}, f, indent=2)
    print(f"Saved results to: {out}")


if __name__ == "__main__":
    env = ENVEvalConfig.from_env()
    main(env)
