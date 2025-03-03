import torch
import folder_paths
import comfy.sd
import comfy.model_management
import comfy.utils
import logging
import gc

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO)

# Enable cuDNN benchmarking if input sizes are fixed
torch.backends.cudnn.benchmark = True

# Global variable for the current device – defaults to "cuda:0" if available
current_device = "cuda:0"


def get_torch_device_patched():
    """
    Returns the appropriate torch.device based on GPU availability and configuration.
    Falls back to CPU if no CUDA devices are available or CPU mode is forced.
    """
    global current_device
    if not torch.cuda.is_available() or \
       comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU:
        logging.info("CUDA not available or CPU mode selected. Using CPU.")
        return torch.device("cpu")
    try:
        # Validate that current_device is in the correct format (e.g., "cuda:0")
        device_index = int(current_device.split(":")[1])
    except (IndexError, ValueError):
        logging.warning("Invalid current_device format; defaulting to cuda:0")
        current_device = "cuda:0"
        return torch.device("cuda:0")
    if device_index >= torch.cuda.device_count():
        logging.warning(f"Requested device cuda:{device_index} not available. Using cuda:0 instead.")
        current_device = "cuda:0"
        return torch.device("cuda:0")
    return torch.device(current_device)


# Patch comfy's device getter to use our custom function
comfy.model_management.get_torch_device = get_torch_device_patched


def load_state_dict_safe(path, device):
    """
    Safely loads a state dict from the given path, moving it directly to the target device.
    Also strips out any DataParallel prefixes.
    """
    try:
        state = torch.load(path, map_location=device)
    except Exception as e:
        logging.error(f"Error loading state dict from {path}: {e}")
        raise e

    # Remove "module." prefix if present (from DataParallel training)
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v.to(device) if isinstance(v, torch.Tensor) else v
    return new_state


def safe_model_load(model_loader_func, *args, **kwargs):
    """
    Wrapper for model loading functions to catch and log exceptions.
    """
    try:
        result = model_loader_func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in loading model via {model_loader_func.__name__}: {e}")
        raise e
    return result


class CheckpointLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, device):
        global current_device
        current_device = device

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        try:
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                map_location=get_torch_device_patched()
            )
        except Exception as e:
            logging.error(f"Failed to load checkpoint '{ckpt_name}' on device {device}: {e}")
            raise e

        # Clean up temporary objects and clear cache
        gc.collect()
        torch.cuda.empty_cache()
        return out[:3]


class UNETLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype, device):
        global current_device
        current_device = device

        dtype = None
        if weight_dtype == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("unet", unet_name)
        try:
            model = comfy.sd.load_unet(unet_path, dtype=dtype, map_location=get_torch_device_patched())
        except Exception as e:
            logging.error(f"Error loading UNet '{unet_name}' on device {device}: {e}")
            raise e

        gc.collect()
        torch.cuda.empty_cache()
        return (model,)


class VAELoaderMultiGPU:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = sdxl_taesd_dec = False
        sd1_taesd_enc = sd1_taesd_dec = False
        sd3_taesd_enc = sd3_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        return vaes

    @staticmethod
    def load_taesd(name, device):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        try:
            encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
            decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))
        except StopIteration as e:
            logging.error(f"Encoder or decoder not found for {name}: {e}")
            raise e

        try:
            enc_path = folder_paths.get_full_path("vae_approx", encoder)
            enc = safe_model_load(comfy.utils.load_torch_file, enc_path)
            for k in enc:
                sd[f"taesd_encoder.{k}"] = enc[k]
            dec_path = folder_paths.get_full_path("vae_approx", decoder)
            dec = safe_model_load(comfy.utils.load_torch_file, dec_path)
            for k in dec:
                sd[f"taesd_decoder.{k}"] = dec[k]
        except Exception as e:
            logging.error(f"Error loading taesd components for {name}: {e}")
            raise e

        # Set scale and shift values on the correct device
        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215, device=device)
            sd["vae_shift"] = torch.tensor(0.0, device=device)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025, device=device)
            sd["vae_shift"] = torch.tensor(0.0, device=device)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305, device=device)
            sd["vae_shift"] = torch.tensor(0.0609, device=device)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "vae_name": (s.vae_list(),),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name, device):
        global current_device
        current_device = device

        if vae_name in ["taesd", "taesdxl", "taesd3"]:
            try:
                sd = self.load_taesd(vae_name, get_torch_device_patched())
            except Exception as e:
                logging.error(f"Error loading taesd for {vae_name}: {e}")
                raise e
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            try:
                sd = safe_model_load(comfy.utils.load_torch_file, vae_path)
            except Exception as e:
                logging.error(f"Error loading VAE '{vae_name}' on device {device}: {e}")
                raise e

        try:
            vae = comfy.sd.VAE(sd=sd)
            vae.to(get_torch_device_patched())
        except Exception as e:
            logging.error(f"Error constructing VAE model: {e}")
            raise e

        gc.collect()
        torch.cuda.empty_cache()
        return (vae,)


class ControlNetLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, device):
        global current_device
        current_device = device

        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        try:
            controlnet = comfy.controlnet.load_controlnet(
                controlnet_path, map_location=get_torch_device_patched()
            )
        except Exception as e:
            logging.error(f"Error loading ControlNet '{control_net_name}' on device {device}: {e}")
            raise e

        gc.collect()
        torch.cuda.empty_cache()
        return (controlnet,)


class CLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio"],),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, device, type="stable_diffusion"):
        global current_device
        current_device = device

        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path("clip", clip_name)
        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                map_location=get_torch_device_patched()
            )
        except Exception as e:
            logging.error(f"Error loading CLIP '{clip_name}' on device {device}: {e}")
            raise e

        gc.collect()
        torch.cuda.empty_cache()
        return (clip,)


class DualCLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "type": (["sdxl", "sd3", "flux"],),
                "device": (devices,),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type, device):
        global current_device
        current_device = device

        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            logging.warning(f"Unrecognized clip type {type}, defaulting to STABLE_DIFFUSION")
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                map_location=get_torch_device_patched()
            )
        except Exception as e:
            logging.error(f"Error loading DualCLIP with '{clip_name1}' and '{clip_name2}' on device {device}: {e}")
            raise e

        gc.collect()
        torch.cuda.empty_cache()
        return (clip,)


# Node mappings for the framework
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderMultiGPU": CheckpointLoaderMultiGPU,
    "UNETLoaderMultiGPU": UNETLoaderMultiGPU,
    "VAELoaderMultiGPU": VAELoaderMultiGPU,
    "ControlNetLoaderMultiGPU": ControlNetLoaderMultiGPU,
    "CLIPLoaderMultiGPU": CLIPLoaderMultiGPU,
    "DualCLIPLoaderMultiGPU": DualCLIPLoaderMultiGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderMultiGPU": "Load Checkpoint (Multi-GPU)",
    "UNETLoaderMultiGPU": "Load Diffusion Model (Multi-GPU)",
    "VAELoaderMultiGPU": "Load VAE (Multi-GPU)",
    "ControlNetLoaderMultiGPU": "Load ControlNet Model (Multi-GPU)",
    "CLIPLoaderMultiGPU": "Load CLIP (Multi-GPU)",
    "DualCLIPLoaderMultiGPU": "DualCLIPLoader (Multi-GPU)",
}
