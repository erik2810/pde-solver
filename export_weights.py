"""Export trained .pth weights to JSON for client-side inference."""

import argparse
import json
from pathlib import Path

import torch

from equations import EQUATIONS
from model import PhysicsInformedNN


def export_model(eq_key: str, model_path: Path, out_dir: Path) -> dict | None:
    eq = EQUATIONS[eq_key]
    try:
        m = PhysicsInformedNN(
            hidden_size=eq.recommended_hidden,
            num_layers=eq.recommended_layers,
        )
        m.load_state_dict(torch.load(str(model_path), weights_only=True))
        m.eval()
    except Exception as e:
        print(f"  skip {eq_key}: {e}")
        return None

    layers = []
    for module in m.net:
        if isinstance(module, torch.nn.Linear):
            layers.append({
                "W": module.weight.detach().tolist(),
                "b": module.bias.detach().tolist(),
            })

    data = {
        "key": eq.key,
        "name": eq.name,
        "description": eq.description,
        "latex_equation": eq.latex_equation,
        "latex_ic": eq.latex_ic,
        "latex_bc": eq.latex_bc,
        "latex_domain": eq.latex_domain,
        "x_min": eq.x_min,
        "x_max": eq.x_max,
        "t_min": eq.t_min,
        "t_max": eq.t_max,
        "y_min": eq.y_min,
        "y_max": eq.y_max,
        "y_label": eq.y_label,
        "params": eq.params,
        "layers": layers,
    }

    out_path = out_dir / f"{eq_key}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"  {eq_key}: {len(layers)} layers, {size_kb:.1f} KB")
    return data


def main():
    parser = argparse.ArgumentParser(description="Export model weights to JSON")
    parser.add_argument(
        "--out-dir", type=str, default="docs/weights",
        help="Output directory for JSON weight files",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing .pth model files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    print("Exporting model weights to JSON...\n")

    manifest = []
    for eq_key in EQUATIONS:
        path = models_dir / f"{eq_key}_model.pth"
        if not path.exists():
            print(f"  skip {eq_key}: no model file")
            continue
        result = export_model(eq_key, path, out_dir)
        if result:
            manifest.append(eq_key)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nExported {len(manifest)} models to {out_dir}/")


if __name__ == "__main__":
    main()
