"""DrivingSceneAnalyzer — one class replacing the three copy-pasted inference loops.

scripts/01_inference.py, 03_eval_inference.py, and 06_lora_eval.py are the
same ~80-line loop with three differences: base model vs LoRA adapter, and
the output filename. This module is that loop, done once, with three
upgrades the copies never got:

- structured records: every reply is parsed into a SceneAnalysis and
  validated; invalid replies get one repair retry with the validation
  errors quoted back to the model (repair_loop=1), and failures are
  recorded as structured error records instead of the old "[错误] ..."
  strings that polluted eval_results.json;
- resume: batch runs skip images already present in the output file, so a
  crashed run continues instead of restarting from zero;
- prompt fidelity: the default prompt is byte-identical to the training
  prompt (04_prepare_training_data.py) — changing it would fight the LoRA
  fine-tune.

torch/transformers/peft are imported lazily inside load_model(), so this
module imports instantly on a CPU-only box (tests import it freely).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .parsing import parse_json_output, parse_text_sections
from .schema import SceneAnalysis, validate

# Byte-identical to the training prompt — do not "improve" the wording
# without retraining; see module docstring.
TRAINING_PROMPT = (
    "请分析这张驾驶场景图片，输出以下信息：\n"
    "1. 车道线数量和类型\n"
    "2. 前方车辆位置和距离估计\n"
    "3. 交通标志/信号灯状态\n"
    "4. 潜在驾驶风险"
)

JSON_PROMPT = (
    "请分析这张驾驶场景图片，只输出一个 JSON 对象，不要输出其他文字，"
    "字段为 lane(车道线数量和类型)、vehicles(前方车辆，格式如 2辆小汽车)、"
    "signs(交通标志/信号灯)、risk(潜在驾驶风险)。"
)

IMAGE_EXTENSIONS = ("*.jpg", "*.png", "*.jpeg")


class DrivingSceneAnalyzer:
    """Configurable analyzer over a base model or a base+LoRA stack.

    generate_fn: pluggable text generator with signature
    ``(image: PIL.Image, prompt: str) -> str``. Production code passes
    None and gets the transformers-based default; tests pass a stub and
    run the full pipeline without torch.
    """

    def __init__(
        self,
        model_path: str | Path,
        lora_path: str | Path | None = None,
        *,
        output_format: str = "text",
        repair_loop: int = 1,
        max_new_tokens: int = 512,
        max_image_size: int = 800,
        generate_fn: Callable[[Any, str], str] | None = None,
    ) -> None:
        if output_format not in ("text", "json"):
            raise ValueError(f"output_format must be 'text' or 'json', got {output_format!r}")
        self.model_path = Path(model_path)
        self.lora_path = Path(lora_path) if lora_path else None
        self.output_format = output_format
        self.repair_loop = repair_loop
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.generate_fn = generate_fn
        self._model = None
        self._processor = None

    @property
    def prompt(self) -> str:
        return TRAINING_PROMPT if self.output_format == "text" else JSON_PROMPT

    def load_model(self) -> None:
        """Load the 4-bit base model (and optional LoRA adapter).

        Kept separate from __init__ so a stubbed analyzer never touches
        torch, and so callers can build the object, inspect the prompt,
        and only then pay the 7GB load.
        """
        if self.generate_fn is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if self.lora_path else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4" if self.lora_path else None,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(str(self.model_path))
        if self.lora_path:
            self._model = PeftModel.from_pretrained(self._model, str(self.lora_path))

    def _generate(self, image: Any, prompt: str) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(image, prompt)
        import torch
        from PIL import Image

        if image is not None and not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        if image is not None:
            image.thumbnail((self.max_image_size, self.max_image_size))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def analyze(self, image: Any, image_name: str = "") -> dict[str, Any]:
        """Analyze one image; always returns a JSON-serializable dict.

        Success shape: SceneAnalysis.to_dict() plus parse_status "ok".
        Failure shape: {"image": ..., "parse_status": "error", "errors": [...]}
        — never an exception, never a "[错误]" string masquerading as data.
        """
        name = image_name or (getattr(image, "name", "") or "")
        prompt = self.prompt
        try:
            reply = self._generate(image, prompt)
        except Exception as exc:  # model/IO failure is a record, not a crash
            return {"image": name, "parse_status": "error", "errors": [f"generation failed: {exc}"], "raw": ""}

        if self.output_format == "json":
            analysis, errors = parse_json_output(reply, image=name)
            attempt = 0
            while errors and attempt < self.repair_loop:
                attempt += 1
                retry_prompt = (
                    f"{prompt}\n\n你上一次的输出不是合法的结构化结果，问题如下：\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n请重新输出。"
                )
                try:
                    reply2 = self._generate(image, retry_prompt)
                except Exception as exc:
                    return {"image": name, "parse_status": "error", "errors": errors + [f"retry generation failed: {exc}"], "raw": reply}
                analysis, errors = parse_json_output(reply2, image=name)
            if errors:
                return {"image": name, "parse_status": "error", "errors": errors, "raw": reply}
            return {**analysis.to_dict(), "parse_status": "ok"}

        analysis = parse_text_sections(reply, image=name)
        errors = validate(analysis)
        if errors:
            # Text mode has no cross-turn repair (the LoRA format is fixed);
            # surface the problems so the evaluator can count them.
            return {**analysis.to_dict(), "parse_status": "partial", "errors": errors}
        return {**analysis.to_dict(), "parse_status": "ok"}

    def analyze_folder(
        self,
        image_folder: str | Path,
        output_file: str | Path,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Batch-analyze a folder with resume support.

        Images already present in output_file (with any parse_status) are
        skipped, so re-running after a crash only processes the remainder —
        the old scripts restarted all 50 images from zero.
        """
        folder = Path(image_folder)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {}
        if out_path.exists():
            try:
                results = json.loads(out_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                results = {}

        image_paths = sorted(p for ext in IMAGE_EXTENSIONS for p in folder.glob(ext))
        if limit is not None:
            image_paths = image_paths[:limit]

        done = 0
        for i, filepath in enumerate(image_paths, 1):
            if filepath.name in results:
                continue
            print(f"[{i}/{len(image_paths)}] {filepath.name}", flush=True)
            record = self.analyze(filepath, image_name=filepath.name)
            results[filepath.name] = record
            done += 1
            if done % 5 == 0 or i == len(image_paths):
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = sum(1 for r in results.values() if r.get("parse_status") == "ok")
        print(f"完成：共 {len(results)} 条记录（本次新增 {done}），结构化成功 {ok} 条")
        return results
