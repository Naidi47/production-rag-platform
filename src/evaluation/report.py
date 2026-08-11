import json
from datetime import datetime, timezone
from pathlib import Path


class Report:
    def __init__(self, metrics: dict[str, float], run_name: str):
        self.metrics = metrics
        self.run_name = run_name

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation Report: {self.run_name}",
            f"Date (UTC): {datetime.now(timezone.utc).isoformat()}",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        lines.extend(f"| {key} | {value:.4f} |" for key, value in self.metrics.items())
        return "\n".join(lines) + "\n"

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.run_name}.json").write_text(
            json.dumps(self.metrics, indent=2), encoding="utf-8"
        )
        (path / f"{self.run_name}.md").write_text(self.to_markdown(), encoding="utf-8")
