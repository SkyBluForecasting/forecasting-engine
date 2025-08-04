import pytest
from unittest.mock import patch
import forecasting_engine.tasks.poll_training_queue as poller


@pytest.fixture
def mock_sqs_client():
    with patch.object(poller, "SQS_CLIENT") as mock_client:
        yield mock_client


@pytest.fixture
def mock_process_message():
    with patch.object(poller, "process_training_queue_message") as mock_proc:
        yield mock_proc


@pytest.mark.usefixtures("mock_sqs_client", "mock_process_message")
class TestProcessOnce:

    def test_process_once_with_messages(self, mock_sqs_client, mock_process_message):
        mock_sqs_client.receive_message.return_value = {
            "Messages": [{"Body": "msg1"}, {"Body": "msg2"}]
        }

        result = poller.process_once()

        assert result is True
        assert mock_process_message.call_count == 2
        mock_process_message.assert_any_call({"Body": "msg1"})
        mock_process_message.assert_any_call({"Body": "msg2"})

    def test_process_once_no_messages(self, mock_sqs_client, mock_process_message):
        mock_sqs_client.receive_message.return_value = {}

        result = poller.process_once()

        assert result is False
        mock_process_message.assert_not_called()


class TestMain:

    def test_main_runs_one_loop(self, monkeypatch):
        """Test main() but stop after first loop to avoid infinite loop."""
        call_count = {"count": 0}

        def fake_process_once():
            call_count["count"] += 1
            if call_count["count"] > 0:
                raise SystemExit  # stop main loop

        monkeypatch.setattr(poller, "process_once", fake_process_once)
        monkeypatch.setattr(poller.time, "sleep", lambda _: None)

        with pytest.raises(SystemExit):
            poller.main()

        assert call_count["count"] == 1
