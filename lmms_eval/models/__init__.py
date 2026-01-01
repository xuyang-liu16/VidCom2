import importlib
import os
import sys
from typing import Literal

from loguru import logger

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
# Configure logger with detailed format including file path, function name, and line number
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)


AVAILABLE_SIMPLE_MODELS = {
    "qwen2_vl": "Qwen2_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen3_vl": "Qwen3_VL",
}

AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "qwen3_vl": "Qwen3_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
}


def get_model(model_name, force_simple: bool = False):
    if model_name not in AVAILABLE_SIMPLE_MODELS and model_name not in AVAILABLE_CHAT_TEMPLATE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    if model_name in AVAILABLE_CHAT_TEMPLATE_MODELS:
        model_type = "chat"
        AVAILABLE_MODELS = AVAILABLE_CHAT_TEMPLATE_MODELS
    else:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    # Override with force_simple if needed, but only if the model exists in AVAILABLE_SIMPLE_MODELS
    if force_simple and model_name in AVAILABLE_SIMPLE_MODELS:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_type}.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        # For plugin users, this will be replaced by chat template model later
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_SIMPLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
