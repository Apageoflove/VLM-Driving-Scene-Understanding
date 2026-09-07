"""Stdlib-only test suite for the vlm_drive package.

Runs with ``python -m unittest discover tests`` on any machine — no torch,
no transformers, no GPU. The analyzer's model interaction is exercised
through a stubbed generate_fn, which is the same seam production code
uses (analyzer._generate).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_drive.analyzer import DrivingSceneAnalyzer, TRAINING_PROMPT
from vlm_drive.evaluator import evaluate, write_report
from vlm_drive.parsing import (
    extract_counts,
    parse_ground_truth,
    parse_json_output,
    parse_text_sections,
    strip_code_fences,
)
from vlm_drive.schema import SceneAnalysis, validate


def trained_format_reply(
    lane: str = "本车所在车道两侧为白色实线，前方车道线清晰。",
    vehicles: str = "前方可见2辆小汽车、1辆卡车。无行人。",
    signs: str = "无交通标志或信号灯。",
    risk: str = "前方车辆较近，需保持车距。",
) -> str:
    return (
        f"1. 车道线：{lane}\n"
        f"2. 车辆：{vehicles}\n"
        f"3. 交通标志/信号灯：{signs}\n"
        f"4. 潜在驾驶风险：{risk}"
    )


class TestExtractCounts(unittest.TestCase):
    def test_multiple_categories(self):
        counts = extract_counts("前方可见2辆小汽车、1辆卡车，右侧1辆摩托车")
        self.assertEqual(counts, {"小汽车": 2, "卡车": 1, "摩托车": 1})

    def test_pedestrian_uses_ge_counter(self):
        self.assertEqual(extract_counts("3个行人"), {"行人": 3})

    def test_repeated_mentions_sum(self):
        self.assertEqual(extract_counts("1辆小汽车 后方又有2辆小汽车"), {"小汽车": 3})

    def test_no_counts(self):
        self.assertEqual(extract_counts("前方无可见车辆"), {})

    def test_empty(self):
        self.assertEqual(extract_counts(""), {})


class TestParseTextSections(unittest.TestCase):
    def test_trained_format_roundtrip(self):
        analysis = parse_text_sections(trained_format_reply(), image="a.jpg")
        self.assertEqual(validate(analysis), [])
        self.assertEqual(analysis.vehicles, {"小汽车": 2, "卡车": 1})
        self.assertEqual(analysis.image, "a.jpg")

    def test_none_variants(self):
        reply = trained_format_reply(vehicles="前方无可见车辆。无行人。")
        analysis = parse_text_sections(reply, image="x.jpg")
        self.assertEqual(analysis.vehicles, {})

    def test_cone_detection(self):
        reply = trained_format_reply(signs="交通锥。")
        analysis = parse_text_sections(reply, image="x.jpg")
        self.assertEqual(analysis.signs, {"交通锥": 1})

    def test_missing_section_reported_not_raised(self):
        analysis = parse_text_sections("1. 车道线：双实线\n2. 车辆：无", image="x.jpg")
        errors = validate(analysis)
        self.assertTrue(any("risk" in e for e in errors), errors)

    def test_section_number_without_dot(self):
        reply = trained_format_reply().replace("3. 交通", "3、交通")
        analysis = parse_text_sections(reply, image="x.jpg")
        self.assertEqual(validate(analysis), [])

    def test_empty_reply(self):
        analysis = parse_text_sections("", image="e.jpg")
        self.assertTrue(validate(analysis))


class TestParseJsonOutput(unittest.TestCase):
    def test_clean_json(self):
        reply = '```json\n{"lane": "双车道", "vehicles": "2辆小汽车", "signs": "无", "risk": "无风险"}\n```'
        analysis, errors = parse_json_output(reply, image="j.jpg")
        self.assertEqual(errors, [])
        self.assertEqual(analysis.vehicles, {"小汽车": 2})

    def test_prose_preamble(self):
        reply = '好的，以下是分析结果：{"lane": "a", "vehicles": {}, "signs": {}, "risk": "b"}'
        analysis, errors = parse_json_output(reply, image="x.jpg")
        self.assertEqual(errors, [])

    def test_trailing_comma_repaired(self):
        reply = '{"lane": "a", "vehicles": {"小汽车": 1}, "signs": {}, "risk": "b",}'
        analysis, errors = parse_json_output(reply, image="x.jpg")
        self.assertEqual(errors, [])
        self.assertEqual(analysis.vehicles, {"小汽车": 1})

    def test_truncated_json_repaired(self):
        reply = '{"lane": "双车道白色实线", "vehicles": {"小汽车": 2, "卡车": 1}, "signs": {"交通锥": 1}, "risk": "保持车'
        analysis, errors = parse_json_output(reply, image="x.jpg")
        # Either fully repaired or errors reported — never an exception.
        self.assertIsInstance(analysis, SceneAnalysis)
        if not errors:
            self.assertEqual(analysis.vehicles.get("小汽车"), 2)

    def test_single_quotes_repaired(self):
        reply = "{'lane': 'a', 'vehicles': {}, 'signs': {}, 'risk': 'b'}"
        analysis, errors = parse_json_output(reply, image="x.jpg")
        self.assertEqual(errors, [])

    def test_dict_counts(self):
        reply = '{"lane":"a","vehicles":{"小汽车":3},"signs":{},"risk":"b"}'
        analysis, errors = parse_json_output(reply, image="x.jpg")
        self.assertEqual(errors, [])
        self.assertEqual(analysis.vehicles["小汽车"], 3)

    def test_garbage_reports_errors(self):
        _, errors = parse_json_output("完全不是 JSON 的一段话", image="g.jpg")
        self.assertTrue(errors)


class TestSchema(unittest.TestCase):
    def test_valid(self):
        a = SceneAnalysis(image="a.jpg", lane="x", vehicles={"小汽车": 1}, signs={}, risk="y")
        self.assertEqual(validate(a), [])

    def test_negative_count_rejected(self):
        a = SceneAnalysis(image="a.jpg", lane="x", vehicles={"小汽车": -1}, signs={}, risk="y")
        self.assertTrue(validate(a))


class TestAnalyzerWithStub(unittest.TestCase):
    def _analyzer(self, replies):
        calls = []

        def generate_fn(image, prompt):
            calls.append(prompt)
            return replies[len(calls) - 1] if len(calls) <= len(replies) else replies[-1]

        return DrivingSceneAnalyzer(
            "unused-model-path",
            output_format="json",
            repair_loop=1,
            generate_fn=generate_fn,
        ), calls

    def test_training_prompt_fidelity(self):
        """The text-mode prompt must stay byte-identical to 04's training prompt."""
        analyzer = DrivingSceneAnalyzer("m", output_format="text")
        self.assertEqual(analyzer.prompt, TRAINING_PROMPT)
        expected = (
            "请分析这张驾驶场景图片，输出以下信息：\n"
            "1. 车道线数量和类型\n"
            "2. 前方车辆位置和距离估计\n"
            "3. 交通标志/信号灯状态\n"
            "4. 潜在驾驶风险"
        )
        self.assertEqual(analyzer.prompt, expected)

    def test_json_ok_first_pass(self):
        good = '{"lane":"a","vehicles":"1辆小汽车","signs":"无","risk":"b"}'
        analyzer, calls = self._analyzer([good])
        record = analyzer.analyze("img", image_name="a.jpg")
        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["vehicles"], {"小汽车": 1})
        self.assertEqual(len(calls), 1)

    def test_repair_loop_recovers(self):
        bad = "对不起我无法输出 JSON"  # first reply invalid
        good = '{"lane":"a","vehicles":{},"signs":{},"risk":"b"}'
        analyzer, calls = self._analyzer([bad, good])
        record = analyzer.analyze("img", image_name="a.jpg")
        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertIn("问题如下", calls[1])

    def test_repair_loop_exhausted_is_error_record(self):
        analyzer, _ = self._analyzer(["垃圾输出", "还是垃圾"])
        record = analyzer.analyze("img", image_name="a.jpg")
        self.assertEqual(record["parse_status"], "error")
        self.assertTrue(record["errors"])

    def test_generation_failure_is_record_not_crash(self):
        def broken(image, prompt):
            raise RuntimeError("CUDA OOM")

        analyzer = DrivingSceneAnalyzer("m", output_format="json", generate_fn=broken)
        record = analyzer.analyze("img", image_name="oom.jpg")
        self.assertEqual(record["parse_status"], "error")
        self.assertIn("generation failed", record["errors"][0])

    def test_analyze_folder_resume(self):
        import tempfile

        good = '{"lane":"a","vehicles":{},"signs":{},"risk":"b"}'
        analyzer, _ = self._analyzer([good])
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "imgs"
            img_dir.mkdir()
            for name in ("a.jpg", "b.jpg"):
                (img_dir / name).write_bytes(b"not-a-real-image")
            out = Path(tmp) / "out.json"
            # First image already present with a finished record.
            out.write_text(json.dumps({"a.jpg": {"parse_status": "ok"}}), encoding="utf-8")
            analyzer.analyze_folder(img_dir, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("a.jpg", data)
            self.assertIn("b.jpg", data)


class TestEvaluator(unittest.TestCase):
    GT = {
        "p1.jpg": trained_format_reply(vehicles="2辆小汽车、1辆卡车。无行人。", signs="无交通标志或信号灯。"),
        "p2.jpg": trained_format_reply(vehicles="前方无可见车辆。3个行人。", signs="交通锥。"),
    }

    def _pred(self, text_by_image):
        return {name: parse_text_sections(text).to_dict() for name, text in text_by_image.items()}

    def test_perfect_predictions_score_full(self):
        preds = self._pred(self.GT)
        result = evaluate(preds, self.GT)
        m = result["metrics"]
        self.assertEqual(m["images_evaluated"], 2)
        self.assertEqual(m["parse_success_rate"], 1.0)
        self.assertEqual(m["vehicle_presence_accuracy"], 1.0)
        self.assertEqual(m["pedestrian_hit_rate"], 1.0)
        self.assertEqual(m["cone_presence_hit_rate"], 1.0)
        self.assertEqual(m["vehicle_count"]["小汽车"]["exact_match_rate"], 1.0)
        self.assertEqual(m["vehicle_count"]["小汽车"]["count_mae"], 0.0)

    def test_wrong_counts_measured(self):
        wrong = dict(self.GT)
        wrong["p1.jpg"] = trained_format_reply(vehicles="1辆小汽车。无行人。", signs="无交通标志或信号灯。")
        result = evaluate(self._pred(wrong), self.GT)
        self.assertEqual(result["metrics"]["vehicle_count"]["小汽车"]["count_mae"], 0.5)
        self.assertEqual(result["metrics"]["vehicle_count"]["小汽车"]["exact_match_rate"], 0.5)

    def test_legacy_raw_text_predictions(self):
        """Old pipelines stored bare strings — evaluator must accept them."""
        result = evaluate(dict(self.GT), self.GT)  # pass GT strings as "predictions"
        self.assertEqual(result["metrics"]["parse_success_rate"], 1.0)

    def test_missing_prediction_reported(self):
        preds = self._pred({"p1.jpg": self.GT["p1.jpg"]})
        result = evaluate(preds, self.GT)
        self.assertEqual(result["metrics"]["images_missing_prediction"], ["p2.jpg"])
        self.assertEqual(result["metrics"]["images_evaluated"], 1)

    def test_report_written(self):
        import tempfile

        result = evaluate(self._pred(self.GT), self.GT)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_report(result, Path(tmp) / "report.md")
            text = out.read_text(encoding="utf-8")
            self.assertIn("结构化成功率", text)
            self.assertIn("p1.jpg", text)
            self.assertIn("小汽车", text)


if __name__ == "__main__":
    unittest.main()


class TestRealLoRAFormat(unittest.TestCase):
    """Regression cases shaped by the actual lora_eval_results.json replies.

    The fine-tuned model drifts from the trained colon format: numbered
    lines with free-form headers, Chinese-numeral counts, and distance
    prose inside the vehicle section. These tests pin the parser's
    handling of each drift.
    """

    def test_freeform_numbered_reply(self):
        reply = (
            "1. 车道线数量为一条车道，车道线类型为黄色实线。\n"
            "2. 前方有三辆汽车，距离分别为：左侧一辆车距离约为50米。\n"
            "3. 图中未显示交通标志或信号灯。\n"
            "4. 驾驶者需小心右侧来车，注意前方车辆的动态变化。"
        )
        analysis = parse_text_sections(reply, image="real.jpg")
        self.assertEqual(validate(analysis), [])
        self.assertEqual(analysis.vehicles, {"汽车": 3})
        self.assertTrue(analysis.risk.startswith("驾驶者需小心"))

    def test_distance_prose_not_counted(self):
        self.assertEqual(extract_counts("左侧一辆车距离约为50米，中间一辆车距离约为70米"), {})

    def test_chinese_numerals(self):
        self.assertEqual(extract_counts("前方有三辆汽车，左侧两辆卡车"), {"汽车": 3, "卡车": 2})

    def test_generic_car_alias_buckets_to_sedan(self):
        from vlm_drive.evaluator import _bucket_vehicles
        self.assertEqual(_bucket_vehicles({"汽车": 3}), {"小汽车": 3})

    def test_real_file_end_to_end_if_present(self):
        """If the real historical results file exists locally, parse-rate >= 0.8."""
        from pathlib import Path
        real = Path(__file__).resolve().parent.parent / "data" / "lora_eval_results.json"
        if not real.exists():
            self.skipTest("local data not present in this checkout")
        records = json.loads(real.read_text(encoding="utf-8"))
        ok = sum(1 for t in records.values() if not validate(parse_text_sections(t, image="x")))
        self.assertGreaterEqual(ok / len(records), 0.8)
