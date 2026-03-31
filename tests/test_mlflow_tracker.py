"""Tests for MLflow tracker."""

from core.mlflow_tracker import NLQTracker


def test_tracker_init(tmp_path):
    """Test tracker initialization."""
    tracker = NLQTracker(tracking_uri=str(tmp_path / "mlruns"))
    assert tracker.step_times == {}


def test_tracker_log_step(tmp_path):
    """Test logging a step."""
    tracker = NLQTracker(tracking_uri=str(tmp_path / "mlruns"))
    with tracker.track_run("test", "claude"):
        tracker.log_step("router", 0.5)
        tracker.log_step("generator", 2.0)

    assert tracker.step_times["router"] == 0.5
    assert tracker.step_times["generator"] == 2.0


def test_tracker_log_results(tmp_path):
    """Test logging results."""
    tracker = NLQTracker(tracking_uri=str(tmp_path / "mlruns"))
    with tracker.track_run("test", "mistral"):
        tracker.log_results(
            route="SQL_QUERY",
            is_valid=True,
            num_rows=10,
            attempts=1,
            sql="SELECT * FROM patients",
        )
