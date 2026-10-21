from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from proact.evaluation.config import load_evaluation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "src/proact/evaluation/assets"
TASKS = {"object_tracking", "state_tracking", "video_capture"}
FAMILIES = {"stateless": 7, "video_llm": 3, "streaming": 2}
CONFIGS_PER_LEVEL = sum(FAMILIES.values()) * len(TASKS)


class EvaluationAssetTests(unittest.TestCase):
    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PROACT_PROJECT_ROOT": str(PROJECT_ROOT),
                "PROACT_EVAL_OUTPUT_ROOT": str(PROJECT_ROOT / "outputs/test-assets"),
                "PROACT_OBJECT_TRACKING_TASK_JSON": str(
                    PROJECT_ROOT / "data/6.full_annotations_no_review/object_tracking.json"
                ),
            }
        )
        return environment

    def test_complete_intervention_and_notification_matrices_load(self) -> None:
        with mock.patch.dict(os.environ, self.environment(), clear=True):
            for root_name, proactive_level in (
                ("configs", "intervention"),
                ("configs_notification", "notification"),
            ):
                root = ASSET_ROOT / root_name
                paths = sorted(root.glob("*/*/*.yaml"))
                self.assertEqual(len(paths), CONFIGS_PER_LEVEL, root_name)
                for task in TASKS:
                    for family, expected_count in FAMILIES.items():
                        family_paths = sorted((root / task / family).glob("*.yaml"))
                        self.assertEqual(
                            len(family_paths),
                            expected_count,
                            (root_name, task, family),
                        )
                        for path in family_paths:
                            with self.subTest(path=path.relative_to(ASSET_ROOT)):
                                config = load_evaluation_config(path)
                                self.assertEqual(config.dataset.dataset, "ProactBenchClips")
                                self.assertEqual(config.task_policy.task_name, task)
                                self.assertEqual(config.proactive_level, proactive_level)
                                self.assertEqual(config.prompt.state_format, "compact")
                                self.assertEqual(
                                    config.prompt_context.completed_tool_intervals.mode,
                                    "none" if proactive_level == "notification" else "last_n",
                                )
                                self.assertTrue(config.parser.unified_schema)
                                self.assertIsNone(config.metrics)

                                if proactive_level == "notification":
                                    self.assertFalse(config.prompt_context.include_active_tools)
                                    self.assertEqual(
                                        config.prompt_context.recent_tool_calls.mode,
                                        "none",
                                    )
                                    self.assertEqual(
                                        config.prompt_context.recent_messages.mode,
                                        "last_n",
                                    )
                                    self.assertEqual(config.prompt_context.recent_messages.n, 5)
                                    self.assertEqual(
                                        config.parser.allow_notification_raw_text,
                                        family == "streaming",
                                    )
                                    # Every family shares the intent-based
                                    # notification prompt; the deprecated
                                    # notification_streaming variant is gone.
                                    self.assertIn(
                                        f"/prompts/notification/{task}.txt",
                                        config.prompt.prompt_path,
                                    )
                                else:
                                    self.assertTrue(config.prompt_context.include_active_tools)
                                    self.assertEqual(
                                        config.prompt_context.recent_tool_calls.mode,
                                        "none",
                                    )
                                    self.assertEqual(
                                        config.prompt_context.recent_tool_calls.n,
                                        5,
                                    )
                                    self.assertEqual(
                                        config.prompt_context.recent_tool_calls.source,
                                        "executed",
                                    )
                                    self.assertEqual(
                                        config.prompt_context.recent_messages.mode,
                                        "none",
                                    )
                                    self.assertFalse(
                                        config.parser.allow_notification_raw_text
                                    )

    def test_intervention_prompts_keep_tool_contract(self) -> None:
        for task in TASKS:
            prompt = (ASSET_ROOT / "prompts" / f"{task}.txt").read_text(
                encoding="utf-8"
            )
            with self.subTest(task=task):
                self.assertIn("Current assistant context", prompt)
                self.assertIn("`current_time_sec`", prompt)
                self.assertIn("`active_tools`", prompt)
                self.assertIn("`completed_tool_intervals`", prompt)
                self.assertNotIn("`recent_tool_calls`", prompt)
                self.assertIn('"message": null', prompt)
                self.assertIn('"tool_calls": []', prompt)
                self.assertNotIn('"act"', prompt)

    def test_notification_prompts_expose_only_message_context_and_output(self) -> None:
        for task in TASKS:
            json_prompt = (
                ASSET_ROOT / "prompts" / "notification" / f"{task}.txt"
            ).read_text(encoding="utf-8")
            with self.subTest(task=task):
                self.assertIn("Current assistant context", json_prompt)
                self.assertIn("`current_time_sec`", json_prompt)
                self.assertIn("`recent_messages`", json_prompt)
                self.assertNotIn("active_tools", json_prompt)
                self.assertNotIn("recent_tool_calls", json_prompt)
                self.assertNotIn("evaluation state", json_prompt.lower())
                self.assertNotIn("tool_calls", json_prompt)
                self.assertNotIn('"act"', json_prompt)
                # The do-nothing literal is written `{"message": null }` in the
                # prompts (stray space); match either spelling so this asserts
                # the contract, not the whitespace.
                self.assertRegex(json_prompt, r'\{"message":\s*null\s*\}')

    def test_notification_streaming_prompt_variant_is_removed(self) -> None:
        self.assertFalse((ASSET_ROOT / "prompts" / "notification_streaming").exists())


if __name__ == "__main__":
    unittest.main()
