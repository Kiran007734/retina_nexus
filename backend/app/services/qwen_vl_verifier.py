"""Optional local Qwen3-VL secondary verifier.

This adapter has no mock path. It reports unavailable until the complete
official model, processor, tokenizer, and a successful real image inference
are present. It never owns or rewrites the primary DR result.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


class QwenUnavailableError(RuntimeError):
    pass


REQUIRED_STAGES = {"no_dr", "mild_npdr", "moderate_npdr", "severe_npdr", "pdr", "ungradable", "unable_to_assess"}
REQUIRED_STATES = {"detected", "not_detected", "suspected", "unable_to_assess"}


class QwenVLVerifier:
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    VERSION = "Qwen3-VL-4B-Instruct"

    def __init__(self, model_path: str | Path, timeout_seconds: int = 120, device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._load_error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        required = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        return self.model_path.is_dir() and all((self.model_path / name).is_file() for name in required)

    def status(self) -> dict[str, Any]:
        if not self.configured:
            return {"status": "NOT_CONFIGURED", "model_id": self.MODEL_ID, "reason": f"Complete local model directory is absent: {self.model_path}"}
        if self._load_error:
            return {"status": "FAILED", "model_id": self.MODEL_ID, "reason": self._load_error}
        if self._model is None:
            return {"status": "NOT_RUN", "model_id": self.MODEL_ID, "reason": "Model preflight/inference has not succeeded."}
        return {"status": "AVAILABLE", "model_id": self.MODEL_ID, "version": self.VERSION, "device": self.device}

    async def verify(self, image_bytes: bytes, context: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._infer(image_bytes, context), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return {"status": "FAILED", "model_id": self.MODEL_ID, "reason": f"Local Qwen inference exceeded {self.timeout_seconds}s."}
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return {"status": "FAILED", "model_id": self.MODEL_ID, "reason": self._load_error}

    async def _infer(self, image_bytes: bytes, context: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            await asyncio.to_thread(self._load)
        image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        prompt = self._prompt(context)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self._model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        output = await asyncio.to_thread(self._model.generate, **inputs, max_new_tokens=512, do_sample=False)
        decoded = self._processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return self.validate_output(self._extract_json(decoded)) | {"model_id": self.MODEL_ID, "version": self.VERSION}

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.configured:
            raise QwenUnavailableError(f"Complete Qwen model files are not present under {self.model_path}.")
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            import torch
        except Exception as exc:
            raise QwenUnavailableError(f"Transformers Qwen3-VL runtime is unavailable: {exc}") from exc
        self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, local_files_only=True)
        self._model.eval()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Qwen response did not contain a JSON object")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Qwen response JSON root is not an object")
        return value

    @staticmethod
    def validate_output(value: dict[str, Any]) -> dict[str, Any]:
        required = {"status", "image_quality", "dr_assessment", "lesion_findings", "macular_assessment", "optic_disc_assessment", "vascular_assessment", "uncertainty", "safety_flags", "explanation"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Qwen response missing fields: {sorted(missing)}")
        if value["status"] not in {"AVAILABLE", "FAILED"}:
            raise ValueError("Qwen status must be AVAILABLE or FAILED")
        stage = value["dr_assessment"].get("stage")
        if stage not in REQUIRED_STAGES:
            raise ValueError(f"Invalid Qwen DR stage: {stage}")
        for name in ("microaneurysms", "hemorrhages", "exudates", "cotton_wool_spots", "neovascularization"):
            if value["lesion_findings"].get(name) not in REQUIRED_STATES:
                raise ValueError(f"Invalid Qwen lesion state: {name}")
        confidence = value["dr_assessment"].get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("Qwen confidence must be numeric in [0, 1]")
        return value

    @staticmethod
    def _prompt(context: dict[str, Any]) -> str:
        return "Return ONLY strict JSON matching the requested schema. You are an independent visual screening-support verifier, not a diagnosis engine. Use unable_to_assess rather than guessing. Do not provide lesion masks, vessel masks, Grad-CAM, patient history, HbA1c, glucose, or unsupported exact measurements. Context: " + json.dumps(context, separators=(",", ":"))
