import random
import sqlite3

import numpy as np
import torch

from fusion_sql.descriptors import DEFAULT_FEATURE_ORDER, ShiftDescriptor
from fusion_sql.evaluation import build_prediction_records
from fusion_sql.meta_learning import FusionSQLMetaLearner, MetaLearningConfig, ShiftDescriptorTask
from fusion_sql.model import FusionSQL


FEATURES = DEFAULT_FEATURE_ORDER


def _descriptor_from_vector(model: str, split_a: str, split_b: str, vector: torch.Tensor) -> ShiftDescriptor:
    return ShiftDescriptor(
        model_name=model,
        split_a=split_a,
        split_b=split_b,
        features={name: float(value) for name, value in zip(FEATURES, vector.tolist())},
    )


def _build_synthetic_tasks(num_tasks: int = 6) -> list[ShiftDescriptorTask]:
    rng = torch.Generator().manual_seed(0)
    tasks: list[ShiftDescriptorTask] = []

    def target_fn(x: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([0.4, -0.25, 0.1, 0.05, -0.02, 0.18])
        return (x * weights).sum(dim=-1, keepdim=True) + 0.1

    for idx in range(num_tasks):
        base = torch.randn((1, len(FEATURES)), generator=rng) * 0.5
        offset = torch.randn((1, len(FEATURES)), generator=rng) * 0.1
        support_vec = base + offset
        query_vec = base - offset
        transfer_vec = base + 2 * offset

        support_label = target_fn(support_vec)
        query_label = target_fn(query_vec)
        transfer_label = target_fn(transfer_vec)

        support_desc = _descriptor_from_vector(f"model{idx}", "meta_train", "meta_val", support_vec.squeeze(0))
        query_desc = _descriptor_from_vector(f"model{idx}", "meta_train", "meta_test", query_vec.squeeze(0))
        transfer_desc = _descriptor_from_vector(f"model{idx}", "meta_train", "dev", transfer_vec.squeeze(0))

        tasks.append(
            ShiftDescriptorTask.from_descriptors(
                model_name=f"model{idx}",
                support=support_desc,
                support_label=float(support_label.item()),
                query=query_desc,
                query_label=float(query_label.item()),
                transfer=transfer_desc,
                transfer_label=float(transfer_label.item()),
                device=torch.device("cpu"),
                feature_order=FEATURES,
            )
        )
    return tasks


def test_fusionsql_forward_matches_functional():
    model = FusionSQL(input_dim=len(FEATURES), dropout=0.0)
    model.eval()
    torch.manual_seed(0)
    inputs = torch.randn(4, len(FEATURES))
    direct = model(inputs)
    params = [p.clone().detach().requires_grad_(True) for p in model.parameter_list()]
    functional = model.functional_forward(inputs, params)
    torch.testing.assert_close(direct, functional)


def test_meta_learner_recovers_synthetic_accuracy():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    tasks = _build_synthetic_tasks()
    model = FusionSQL(input_dim=len(FEATURES), dropout=0.0)
    cfg = MetaLearningConfig(
        inner_lr=0.1,
        outer_lr=1e-2,
        inner_steps=3,
        tasks_per_batch=3,
        num_epochs=180,
        eval_inner_steps=5,
        device="cpu",
    )
    learner = FusionSQLMetaLearner(model, cfg)
    learner.meta_train(tasks)
    transfer_results = learner.evaluate_transfer(tasks)
    mean_mae = np.mean([entry["mae"] for entry in transfer_results])
    assert mean_mae < 0.05, f"Expected MAE < 0.05, got {mean_mae:.4f}"


def test_build_prediction_records_execution(tmp_path):
    db_file = tmp_path / "demo.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE student(age INT)")
    conn.executemany("INSERT INTO student(age) VALUES (?)", [(18,), (19,), (20,)])
    conn.commit()
    conn.close()

    samples = [
        {
            "db_id": "demo",
            "db_path": str(db_file),
            "question": "Sum ages",
            "sql": "SELECT SUM(age) FROM student",
        }
    ]
    preds = ["SELECT SUM(age) FROM student"]
    metrics, records = build_prediction_records(samples, preds, [0])
    assert metrics["execution_accuracy"] == 1.0
    assert records[0]["execution_correct"] is True
