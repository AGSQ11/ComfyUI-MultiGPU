# ComfyUI-MultiGPU

### Experimental nodes for using multiple GPUs in a single ComfyUI workflow.

This extension adds new loader nodes that allow you to specify the GPU to use for each model. It monkey-patches the memory management of ComfyUI in a hacky way to achieve multi-GPU support. **Note:** This solution is experimental and not a comprehensive or thoroughly tested implementation. Use at your own risk.

> **Important:** This extension does not add true parallelism. Workflow steps are still executed sequentially, but models remain on their specified GPUs to avoid repeated loading/unloading of VRAM. The main benefit is reduced overhead when switching between large models.

## Installation

Clone this repository inside the `ComfyUI/custom_nodes/` directory.

## Nodes

![](examples/nodes.png)

The extension adds new loader nodes corresponding to the default ones. They function similarly to the originals but include an extra `device` parameter to choose the GPU.

- `CheckpointLoaderMultiGPU`
- `CLIPLoaderMultiGPU`
- `ControlNetLoaderMultiGPU`
- `DualCLIPLoaderMultiGPU`
- `UNETLoaderMultiGPU`
- `VAELoaderMultiGPU`

## Example Workflows

All workflows have been tested on a 2×3090 setup.

### Loading Two SDXL Checkpoints on Different GPUs

- [Download](examples/sdxl_2gpu.json)

This workflow loads two SDXL checkpoints on two different GPUs: GPU 0 for the first and GPU 1 for the second.

### Split FLUX.1-dev Across Two GPUs

- [Download](examples/flux1dev_2gpu.json)

This workflow loads a FLUX.1-dev model split across two GPUs. The UNet model is loaded on GPU 0 while the text encoders and VAE are loaded on GPU 1.

### FLUX.1-dev and SDXL in the Same Workflow

- [Download](examples/flux1dev_sdxl_2gpu.json)

This workflow loads a FLUX.1-dev model and an SDXL model simultaneously. The FLUX.1-dev model is loaded on GPU 0, and the SDXL model is loaded on GPU 1.

## Code Improvements

The updated loader code includes several modifications and enhancements (all modifications are credited to user [AGSQ11](https://github.com/AGSQ11)):

- **Robust Device Checking:** Verifies GPU availability and falls back to CPU or a default GPU if needed.
- **Direct Weight Loading:** Uses the `map_location` parameter to load checkpoints directly on the target device.
- **Error Handling:** Incorporates try/except blocks and logging to catch and report issues during model loading.
- **Memory Management:** Cleans up resources and empties GPU caches after model loads.
- **Modular and Maintainable Code:** Uses helper functions for common tasks and consistent device management, making it easier to maintain and extend.

## Support

If you encounter problems, please [open an issue](https://github.com/neuratech-ai/ComfyUI-MultiGPU/issues/new). Attach your workflow details if possible.

## Credits

Made by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov).

Modifications and improvements in the code implementation are contributed by user [AGSQ11](https://github.com/AGSQ11).

For business inquiries, email [sales@neuratech.io](mailto:sales@neuratech.io) or visit [our website](https://neuratech.io/).
