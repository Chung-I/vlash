"""Resume bookkeeping for the per-episode-resumable eval client.

Pure logic only (no LIBERO/multiprocessing) -- covers the skip decision,
next-start-episode computation, in-order episode recording, and the
atomic-write/reload round trip that backs crash recovery.
"""
import json

import pytest

from benchmarks.libero.eval_client import (
    _load_prev_results,
    _write_results,
    record_episode,
    resume_start_ep,
    task_done,
)


def test_task_done_false_when_missing():
    assert task_done(None, num_trials=50) is False


def test_task_done_false_when_partial():
    assert task_done({"successes": 3, "trials": 3, "ep_results": [True] * 3}, num_trials=50) is False


def test_task_done_true_when_full():
    assert task_done({"successes": 40, "trials": 50}, num_trials=50) is True


def test_task_done_true_for_old_schema_without_ep_results():
    # Backward compatibility: a task fully completed by the pre-resume
    # client has "successes"/"trials" but no "ep_results" key at all.
    assert task_done({"successes": 45, "trials": 50}, num_trials=50) is True


def test_resume_start_ep_zero_for_missing():
    assert resume_start_ep(None) == 0
    assert resume_start_ep({}) == 0


def test_resume_start_ep_matches_ep_results_length():
    task_result = {"successes": 2, "trials": 3, "ep_results": [True, False, True]}
    assert resume_start_ep(task_result) == 3


def test_record_episode_appends_and_recomputes():
    task_result = {"successes": 0, "trials": 0, "ep_results": []}
    record_episode(task_result, 0, True)
    record_episode(task_result, 1, False)
    record_episode(task_result, 2, True)
    assert task_result["ep_results"] == [True, False, True]
    assert task_result["successes"] == 2
    assert task_result["trials"] == 3


def test_record_episode_rejects_out_of_order():
    task_result = {"successes": 0, "trials": 0, "ep_results": []}
    record_episode(task_result, 0, True)
    with pytest.raises(ValueError):
        record_episode(task_result, 5, True)


def test_write_then_load_round_trip(tmp_path):
    out_path = tmp_path / "libero_spatial_d0.json"
    results = {
        "task_a": {"successes": 1, "trials": 1, "ep_results": [True]},
    }
    out = _write_results(out_path, "libero_spatial", 0, results)
    assert out["success_rate"] == 1.0
    assert not (tmp_path / "libero_spatial_d0.json.tmp").exists()  # tmp cleaned up via os.replace

    on_disk = json.loads(out_path.read_text())
    assert on_disk["suite"] == "libero_spatial"
    assert on_disk["delay"] == 0
    assert on_disk["per_task"] == results

    reloaded = _load_prev_results(out_path)
    assert reloaded == results


def test_write_results_handles_zero_trials_without_div_by_zero(tmp_path):
    out_path = tmp_path / "empty.json"
    out = _write_results(out_path, "libero_object", 3, {})
    assert out["success_rate"] == 0.0


def test_load_prev_results_missing_file_returns_empty(tmp_path):
    assert _load_prev_results(tmp_path / "does_not_exist.json") == {}


def test_load_prev_results_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json")
    assert _load_prev_results(p) == {}


def test_resume_skip_then_continue_end_to_end(tmp_path):
    """Simulates a shard resume: one task already done, one partially done."""
    out_path = tmp_path / "shard.json"
    _write_results(
        out_path,
        "libero_10",
        1,
        {
            "task_0": {"successes": 50, "trials": 50, "ep_results": [True] * 50},
            "task_1": {"successes": 2, "trials": 2, "ep_results": [True, False]},
        },
    )

    results = _load_prev_results(out_path)
    assert task_done(results["task_0"], num_trials=50) is True
    assert task_done(results["task_1"], num_trials=50) is False
    assert resume_start_ep(results["task_1"]) == 2

    record_episode(results["task_1"], 2, True)
    _write_results(out_path, "libero_10", 1, results)

    reloaded = _load_prev_results(out_path)
    assert reloaded["task_1"]["trials"] == 3
    assert reloaded["task_1"]["successes"] == 2
