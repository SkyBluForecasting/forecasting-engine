import pytest
from unittest.mock import Mock
import forecasting_engine.services.forecast_poller.main as poller


# ----------------------------
# Tests for poll_sqs()
# ----------------------------
class TestPollSQS:

    def test_poll_sqs_with_messages(self, monkeypatch, caplog):
        # Fake SQS client returning 2 messages
        fake_sqs = Mock()
        fake_sqs.receive_message.return_value = {
            "Messages": [{"Body": "msg1"}, {"Body": "msg2"}]
        }
        monkeypatch.setattr(poller, "SQS_CLIENT", fake_sqs)

        # Patch process_measurement_queue_message: fail first, succeed second
        def fake_process(msg):
            return msg["Body"] != "msg1"

        monkeypatch.setattr(poller, "process_measurement_queue_message", fake_process)

        with caplog.at_level("WARNING"):
            result = poller.poll_sqs()

        assert result == 2
        assert any(
            "Message was not processed successfully, leaving in queue." in m
            for m in caplog.messages
        )

    def test_poll_sqs_no_messages(self, monkeypatch):
        fake_sqs = Mock()
        fake_sqs.receive_message.return_value = {}
        monkeypatch.setattr(poller, "SQS_CLIENT", fake_sqs)

        monkeypatch.setattr(
            poller, "process_measurement_queue_message", lambda msg: True
        )

        result = poller.poll_sqs()
        assert result == 0

    def test_poll_sqs_raises_exception(self, monkeypatch, caplog):
        fake_sqs = Mock()
        fake_sqs.receive_message.side_effect = RuntimeError("SQS failure")
        monkeypatch.setattr(poller, "SQS_CLIENT", fake_sqs)

        with caplog.at_level("ERROR"):
            result = poller.poll_sqs()

        assert result == 0
        assert any("Error polling SQS: SQS failure" in msg for msg in caplog.messages)


# ----------------------------
# Tests for main()
# ----------------------------
class TestMain:

    def test_main_runs_one_loop(self, monkeypatch):
        call_count = {"count": 0}

        def fake_poll_sqs():
            call_count["count"] += 1
            if call_count["count"] > 0:
                raise SystemExit
            return 0

        monkeypatch.setattr(poller, "poll_sqs", fake_poll_sqs)
        monkeypatch.setattr(poller.time, "sleep", lambda _: None)

        with pytest.raises(SystemExit):
            poller.main()

        assert call_count["count"] == 1

    def test_main_full_coverage_branches(self, monkeypatch):
        call_log = {"calls": []}

        def fake_poll_sqs_zero():
            call_log["calls"].append("zero")
            return 0

        def fake_poll_sqs_nonzero():
            call_log["calls"].append("nonzero")
            return 3

        monkeypatch.setattr(poller.time, "sleep", lambda _: None)
        call_sequence = [
            fake_poll_sqs_zero,
            fake_poll_sqs_nonzero,
            lambda: (_ for _ in ()).throw(SystemExit),
        ]
        monkeypatch.setattr(poller, "poll_sqs", lambda: call_sequence.pop(0)())

        with pytest.raises(SystemExit):
            poller.main()

        assert call_log["calls"] == ["zero", "nonzero"]
