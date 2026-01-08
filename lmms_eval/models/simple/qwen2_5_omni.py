import base64
import os
import sys
import time
from pathlib import Path
from io import BytesIO
from typing import List, Optional, Tuple, Union

import audioread
import av
import decord
import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

_LOCAL_QWEN_OMNI_UTILS = Path(__file__).resolve().parents[3] / "qwen-omni-utils" / "src"
if _LOCAL_QWEN_OMNI_UTILS.exists():
    sys.path.insert(0, str(_LOCAL_QWEN_OMNI_UTILS))

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")


def _format_efficiency_table(rows, title="Efficiency Analysis"):
    headers = ("Metric", "Value")
    col_widths = [
        max(len(headers[0]), max(len(row[0]) for row in rows)),
        max(len(headers[1]), max(len(row[1]) for row in rows)),
    ]
    border = f"+{'-' * (col_widths[0] + 2)}+{'-' * (col_widths[1] + 2)}+"
    header_line = f"| {headers[0]:<{col_widths[0]}} | {headers[1]:<{col_widths[1]}} |"
    lines = [title, border, header_line, border]
    for metric, value in rows:
        lines.append(f"| {metric:<{col_widths[0]}} | {value:<{col_widths[1]}} |")
    lines.append(border)
    return "\n".join(lines)


@register_model("qwen2_5_omni")
class Qwen2_5_Omni(lmms):
    """
    Qwen2.5-Omni-7B
    "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"

    For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on your running tasks.
    https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "eager",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        Qwen2_5OmniForConditionalGeneration._tp_plan = [] if Qwen2_5OmniForConditionalGeneration._tp_plan is None else Qwen2_5OmniForConditionalGeneration._tp_plan
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation=attn_implementation).eval()
        if os.getenv("COMPRESSOR") == "vidcom2":
            import types

            from token_compressor.vidcom2.models.qwen2_5_omni import Qwen2_5_OmniThinker_forward

            self._model.thinker.forward = types.MethodType(Qwen2_5_OmniThinker_forward, self._model.thinker)
            eval_logger.success("[VidCom2] Successfully integrated VidCom2 with Qwen2.5-Omni.")
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._model.disable_talker()
        self.system_prompt = system_prompt
        self.total_cuda_time = 0
        self.max_mem = 0
        self.token_stats_enabled = os.getenv("TOKEN_STATS", "0") == "1"
        self.token_stats_case = os.getenv("TOKEN_STATS_CASE", "0") == "1"
        if self.token_stats_case:
            self.token_stats_enabled = True
        self.token_stats = {
            "video": 0,
            "audio": 0,
            "image": 0,
            "text": 0,
            "total": 0,
            "samples": 0,
        }

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        """
        Resample the audio to the target sample rate.
        """
        if current_sample_rate != 16000:  # The sample rate for Qwen2.5-Omni is 16kHz
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False  # Flag to check whether we are using video or not
        wall_start = time.time()

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on tasks.
            # https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
            message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        current_use_audio = self._check_if_video_has_audio(visual)
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps, img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    # Fixed code for audio messages
                    elif isinstance(visual, dict):  # Single audio
                        current_use_audio = True
                        audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                        single_message = {"role": "user", "content": []}
                        for i in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[i]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
                        current_use_audio = True
                        for i, v in enumerate(visual):
                            audio = self.resample_audio(v["array"], v["sampling_rate"])
                            audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                            single_message = {"role": "user", "content": []}
                            for j in range(len(audio_splits)):
                                single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                            single_message["content"].append({"type": "text", "text": context})
                            message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            use_audio_for_mm = current_use_audio and not self.use_custom_video_loader
            if current_use_audio and self.use_custom_video_loader and self.rank == 0:
                eval_logger.warning("Audio-in-video is disabled when using the custom video loader.")
            audios, images, videos = process_mm_info(message, use_audio_in_video=use_audio_for_mm)
            processor_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "use_audio_in_video": use_audio_for_mm,
            }
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, **processor_kwargs)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if self.token_stats_enabled:
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                mask = attention_mask.bool()
                video_token_id = self.tokenizer.convert_tokens_to_ids(self.processor.video_token)
                audio_token_id = self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
                image_token_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
                total_tokens = mask.sum(dim=1)
                video_tokens = ((input_ids == video_token_id) & mask).sum(dim=1)
                audio_tokens = ((input_ids == audio_token_id) & mask).sum(dim=1)
                image_tokens = ((input_ids == image_token_id) & mask).sum(dim=1)
                text_tokens = total_tokens - video_tokens - audio_tokens - image_tokens
                feature_attention_mask = inputs.get("feature_attention_mask")
                if feature_attention_mask is not None:
                    audio_feat_tokens = feature_attention_mask.sum(dim=1).to(audio_tokens.device)
                    audio_tokens = torch.where(audio_tokens == 0, audio_feat_tokens, audio_tokens)
                self.token_stats["video"] += int(video_tokens.sum().item())
                self.token_stats["audio"] += int(audio_tokens.sum().item())
                self.token_stats["image"] += int(image_tokens.sum().item())
                self.token_stats["text"] += int(text_tokens.sum().item())
                self.token_stats["total"] += int(total_tokens.sum().item())
                self.token_stats["samples"] += int(input_ids.shape[0])
                if self.token_stats_case and self.rank == 0:
                    for i in range(input_ids.shape[0]):
                        print(
                            "[TokenStatsCase] "
                            f"task={task} doc_id={doc_id[i]} "
                            f"video={int(video_tokens[i].item())} "
                            f"audio={int(audio_tokens[i].item())} "
                            f"text={int(text_tokens[i].item())}",
                            flush=True,
                        )

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            try:
                torch.cuda.reset_peak_memory_stats()
                gen_start_event = torch.cuda.Event(enable_timing=True)
                gen_end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                gen_start_event.record()
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
                gen_end_event.record()
                torch.cuda.synchronize()
                gen_time = gen_start_event.elapsed_time(gen_end_event) / 1000.0
                gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.total_cuda_time += gen_time
                self.max_mem = max(gen_max_mem, self.max_mem)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            if os.getenv("VIDCOM_TOKEN_STATS_CASE", "0") == "1" and os.getenv("COMPRESSOR") == "vidcom2":
                thinker = getattr(self.model, "thinker", None)
                last_stats = getattr(thinker, "_vidcom2_last_stats", None)
                if last_stats and self.rank == 0:
                    for i, item in enumerate(last_stats[: len(doc_id)]):
                        pre = item["pre"]
                        post = item["post"]
                        print(
                            "[VidCom2TokenStatsCase] "
                            f"task={task} doc_id={doc_id[i]} "
                            f"pre_video={pre['video']} pre_audio={pre['audio']} pre_text={pre['text']} "
                            f"post_video={post['video']} post_audio={post['audio']} post_text={post['text']}",
                            flush=True,
                        )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        if os.getenv("VIDCOM_TOKEN_STATS", "0") == "1":
            self._write_vidcom2_stats()
        self._write_efficiency_and_token_stats(wall_start)
        return res

    def _write_vidcom2_stats(self) -> None:
        thinker = getattr(self.model, "thinker", None)
        stats = getattr(thinker, "_vidcom2_stats", None)
        if stats is None:
            return
        values = torch.tensor(
            [
                stats["pre"]["video"],
                stats["pre"]["audio"],
                stats["pre"]["text"],
                stats["post"]["video"],
                stats["post"]["audio"],
                stats["post"]["text"],
                stats["samples"],
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            if self.rank != 0:
                return
        pre_video, pre_audio, pre_text, post_video, post_audio, post_text, samples = [float(v.item()) for v in values]
        samples = max(samples, 1.0)
        payload = {
            "samples": int(samples),
            "pre_avg_video_tokens": pre_video / samples,
            "pre_avg_audio_tokens": pre_audio / samples,
            "pre_avg_text_tokens": pre_text / samples,
            "post_avg_video_tokens": post_video / samples,
            "post_avg_audio_tokens": post_audio / samples,
            "post_avg_text_tokens": post_text / samples,
        }
        payload["pre_avg_total_tokens"] = (
            payload["pre_avg_video_tokens"] + payload["pre_avg_audio_tokens"] + payload["pre_avg_text_tokens"]
        )
        payload["post_avg_total_tokens"] = (
            payload["post_avg_video_tokens"] + payload["post_avg_audio_tokens"] + payload["post_avg_text_tokens"]
        )
        table = _format_efficiency_table(
            [
                ("samples", f"{payload['samples']}"),
                ("pre_avg_video_tokens", f"{payload['pre_avg_video_tokens']:.3f}"),
                ("pre_avg_audio_tokens", f"{payload['pre_avg_audio_tokens']:.3f}"),
                ("pre_avg_text_tokens", f"{payload['pre_avg_text_tokens']:.3f}"),
                ("pre_avg_total_tokens", f"{payload['pre_avg_total_tokens']:.3f}"),
                ("post_avg_video_tokens", f"{payload['post_avg_video_tokens']:.3f}"),
                ("post_avg_audio_tokens", f"{payload['post_avg_audio_tokens']:.3f}"),
                ("post_avg_text_tokens", f"{payload['post_avg_text_tokens']:.3f}"),
                ("post_avg_total_tokens", f"{payload['post_avg_total_tokens']:.3f}"),
            ],
            title="VidCom2 Token Stats",
        )
        print(table, flush=True)

    def _write_token_stats(self) -> None:
        stats = self.token_stats
        values = torch.tensor(
            [
                stats["video"],
                stats["audio"],
                stats["image"],
                stats["text"],
                stats["total"],
                stats["samples"],
            ],
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            if self.rank != 0:
                return
        video, audio, image, text, total, samples = [int(v.item()) for v in values]
        samples = max(samples, 1)
        payload = {
            "samples": samples,
            "avg_video_tokens": video / samples,
            "avg_audio_tokens": audio / samples,
            "avg_text_tokens": text / samples,
        }
        payload["avg_total_tokens"] = (
            payload["avg_video_tokens"] + payload["avg_audio_tokens"] + payload["avg_text_tokens"]
        )
        table = _format_efficiency_table(
            [
                ("samples", f"{payload['samples']}"),
                ("avg_video_tokens", f"{payload['avg_video_tokens']:.3f}"),
                ("avg_audio_tokens", f"{payload['avg_audio_tokens']:.3f}"),
                ("avg_text_tokens", f"{payload['avg_text_tokens']:.3f}"),
                ("avg_total_tokens", f"{payload['avg_total_tokens']:.3f}"),
            ],
            title="Token Stats",
        )
        print(table, flush=True)

    def _write_efficiency_and_token_stats(self, wall_start: float) -> None:
        wall_time = time.time() - wall_start
        if self.rank == 0:
            eff_rows = [
                ("LLM_time_s", f"{self.total_cuda_time:.3f}"),
                ("Total_time_s", f"{wall_time:.3f}"),
                ("Peak_mem_MB", f"{self.max_mem:.1f}"),
            ]
            table = _format_efficiency_table(eff_rows, title="Efficiency Analysis")
            print(table, flush=True)

        if self.token_stats_enabled:
            self._write_token_stats()

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
