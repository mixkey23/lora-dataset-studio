# Demo training presets

The `*_Fine_Tuning_Demo.toml` presets select full-DiT fine-tuning for FLUX.2, FLUX.2 Klein, Krea 2, and Z-Image. They use BF16 trainable weights, Adafactor fused backward, gradient checkpointing, and architecture-appropriate block swap settings.

Ideogram 4 full fine-tuning cannot use the official FP8/NVFP4 release. Its preset deliberately leaves `dit` empty; select a plain conditional FP32, FP16, or BF16 checkpoint before training. The existing `Ideogram_4_LoRA_Demo.toml` remains ready for the official FP8 model.

The installed Musubi backend does not provide full-DiT Wan training. `Wan_2.1_LoRA_Fine_Tuning_Demo.toml` therefore uses the supported Wan 2.1 LoRA fine-tuning path instead of presenting a nonfunctional full-model option.
