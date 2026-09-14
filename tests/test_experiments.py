from __future__ import annotations

import numpy as np

from module3.experiments import EXPERIMENT_MATRIX, rows_to_csv, rows_to_markdown, run_experiment_matrix


def test_experiment_matrix_runs_real_metrics_for_every_row() -> None:
    y, x = np.mgrid[0:18, 0:22]
    image = (10 * np.sin(x / 3) + 12 * np.cos(y / 4) + x * y).astype(np.float64)
    rows = run_experiment_matrix(image)
    assert len(rows) == len(EXPERIMENT_MATRIX)
    assert {row.experiment for row in rows} == {spec.experiment for spec in EXPERIMENT_MATRIX}
    assert all(row.mae < 1e-9 for row in rows)
    assert all(row.mse < 1e-15 for row in rows)


def test_results_serializers_include_required_fields() -> None:
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    rows = run_experiment_matrix(image)
    csv = rows_to_csv(rows)
    markdown = rows_to_markdown(rows, image_name="sample.png")
    assert "experiment,filter,kernel_size,mae,mse" in csv
    assert "Average Blur" in csv
    assert "| Experiment | Filter | Kernel Size |" in markdown
    assert "sample.png" in markdown
