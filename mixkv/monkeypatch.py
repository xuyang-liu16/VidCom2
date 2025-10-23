from importlib.metadata import version
import transformers

from mixkv.mistral_model import mistral_flash_attn2_forward_AdaKV, mistral_flash_attn2_forward_MixSparseMM,  mistral_flash_attn2_forward_PyramidKV, mistral_flash_attn2_forward_SnapKV, \
                                   mistral_flash_attn2_forward_SparseMM, mistral_flash_attn2_forward_Mask
from mixkv.mistral_model import prepare_inputs_for_generation_mistral_new, adaptive_MistralModel_forward

from mixkv.qwen2_self import flash_attn_forward_adakv, flash_attn_forward_snapkv, qwen2_forward_adakv,flash_attn_forward_pyramidkv
from mixkv.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_MixSparseMM, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from mixkv.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward





def replace_mistral(method):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SparseMM
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_MixSparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new



def replace_qwen(method):
    if method == 'snapkv':
        print("Using SnapKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_Mask
    
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_MixSparseMM
    if method not in ["fullkv"]:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen

def replace_internvl(method):
    if method=="adakv":
        print("Using Adakv")
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward=qwen2_forward_adakv
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_adakv
    elif method=="snapkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_snapkv
        print("Using Snapkv")
    elif method=="pyramidkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_pyramidkv
        print("Using pyramidkv")


