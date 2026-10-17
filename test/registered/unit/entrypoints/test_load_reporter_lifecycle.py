"""HTTP parent ownership and failure cleanup for the independent reporter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class TestLoadReporterHttpParent(CustomTestCase):
    def test_http_parent_closes_reporter_and_native_resources_on_exit(self):
        for failure in (None, "reporter", "http"):
            with self.subTest(failure=failure):
                args = ServerArgs(model_path="dummy", tokenizer_worker_num=2)
                publish(args, role="tokenizer")
                previous_state = http_server.get_global_state()
                manager = SimpleNamespace(startup_time=0, socket_mapping=Mock())
                reporter, shared_args = Mock(), Mock()
                if failure == "reporter":
                    reporter.start.side_effect = RuntimeError("reporter failed")
                try:
                    with (
                        patch.object(
                            http_server,
                            "write_data_for_multi_tokenizer",
                            return_value=shared_args,
                        ),
                        patch.object(http_server, "set_uvicorn_logging_configs"),
                        patch.object(
                            http_server, "init_load_reporter", return_value=reporter
                        ) as create_reporter,
                        patch.dict(
                            http_server.uvicorn.config.LOGGING_CONFIG["loggers"]
                        ),
                        patch.object(http_server.uvicorn, "run") as serve,
                    ):
                        if failure == "http":
                            serve.side_effect = RuntimeError("http failed")

                        def run():
                            http_server._setup_and_run_http_server(
                                server_args=args,
                                tokenizer_manager=manager,
                                template_manager=None,
                                port_args=None,
                                scheduler_infos=[{}],
                                subprocess_watchdog=None,
                            )

                        if failure:
                            with self.assertRaisesRegex(
                                RuntimeError, f"{failure} failed"
                            ):
                                run()
                        else:
                            run()
                        create_reporter.assert_called_once_with(http_parent=True)
                        reporter.start.assert_called_once_with()
                        reporter.close.assert_called_once_with()
                        shared_args.unlink.assert_called_once_with()
                        manager.socket_mapping.clear_all_sockets.assert_called_once_with()
                        self.assertEqual(serve.call_count, failure != "reporter")
                finally:
                    http_server._global_state = previous_state
                    reset_context()


if __name__ == "__main__":
    unittest.main()
