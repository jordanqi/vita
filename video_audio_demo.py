# # 这是video_audio_memory.py
# import argparse
# import os
# import time
#
# import numpy as np
# import torch
# from PIL import Image
#
# # from gptq import GPTQ
# # from .quant import Quantizer
#
#
# from decord import VideoReader, cpu
# from vita.constants import (
#     DEFAULT_AUDIO_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_VIDEO_TOKEN,
#     IGNORE_INDEX,
#     IMAGE_TOKEN_INDEX,
#     MAX_IMAGE_LENGTH,
# )
# from vita.conversation import SeparatorStyle, conv_templates
# from vita.model.builder import load_pretrained_model
# from vita.util.mm_utils import (
#     KeywordsStoppingCriteria,
#     get_model_name_from_path,
#     tokenizer_image_audio_token,
#     tokenizer_image_token,
# )
# from vita.util.utils import disable_torch_init
#
#
# def _get_rawvideo_dec(
#     video_path,
#     image_processor,
#     max_frames=MAX_IMAGE_LENGTH,
#     min_frames=4,
#     image_resolution=384,
#     video_framerate=1,
#     s=None,
#     e=None,
#     image_aspect_ratio="pad",
# ):
#     # speed up video decode via decord.
#
#     if s is None:
#         start_time, end_time = None, None
#     else:
#         start_time = int(s)
#         end_time = int(e)
#         start_time = start_time if start_time >= 0.0 else 0.0
#         end_time = end_time if end_time >= 0.0 else 0.0
#         if start_time > end_time:
#             start_time, end_time = end_time, start_time
#         elif start_time == end_time:
#             end_time = start_time + 1
#
#     if os.path.exists(video_path):
#         vreader = VideoReader(video_path, ctx=cpu(0))
#     else:
#         print(video_path)
#         raise FileNotFoundError
#
#     fps = vreader.get_avg_fps()
#     f_start = 0 if start_time is None else int(start_time * fps)
#     f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
#     num_frames = f_end - f_start + 1
#     if num_frames > 0:
#         # T x 3 x H x W
#         sample_fps = int(video_framerate)
#         t_stride = int(round(float(fps) / sample_fps))
#
#         all_pos = list(range(f_start, f_end + 1, t_stride))
#         if len(all_pos) > max_frames:
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
#             ]
#         elif len(all_pos) < min_frames:
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
#             ]
#         else:
#             sample_pos = all_pos
#
#         patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
#
#         if image_aspect_ratio == "pad":
#
#             def expand2square(pil_img, background_color):
#                 width, height = pil_img.size
#                 if width == height:
#                     return pil_img
#                 elif width > height:
#                     result = Image.new(pil_img.mode, (width, width), background_color)
#                     result.paste(pil_img, (0, (width - height) // 2))
#                     return result
#                 else:
#                     result = Image.new(pil_img.mode, (height, height), background_color)
#                     result.paste(pil_img, ((height - width) // 2, 0))
#                     return result
#
#             patch_images = [
#                 expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
#                 for i in patch_images
#             ]
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#         else:
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#
#         patch_images = torch.stack(patch_images)
#         slice_len = patch_images.shape[0]
#
#         return patch_images, slice_len
#     else:
#         print("video path: {} error.".format(video_path))
#
#
# if __name__ == "__main__":
#     # Initialize the parser
#     parser = argparse.ArgumentParser(description="Process model and video paths.")
#
#     # Add arguments
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
#     parser.add_argument("--model_base", type=str, default=None)
#     parser.add_argument("--video_path", type=str, default=None)
#     parser.add_argument("--image_path", type=str, default=None)
#     parser.add_argument("--audio_path", type=str, default=None)
#     parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
#     parser.add_argument("--conv_mode", type=str, default="mixtral_two")
#     parser.add_argument("--question", type=str, default="")
#     parser.add_argument("--frameCat", action='store_true')
#     # 在 parser.add_argument 部分添加
#     parser.add_argument("--disable_quant", type=int, default=0,
#                         help="Enable quantization (0=enable, 1=disable)")
#
#     # Parse the arguments
#     args = parser.parse_args()
#
#     # Assign arguments to variables
#     model_path = args.model_path
#     model_base = args.model_base
#     video_path = args.video_path
#     image_path = args.image_path
#     audio_path = args.audio_path
#     qs = args.question
#     assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
#     conv_mode = args.conv_mode
#
#     if args.frameCat:
#         from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
#     else:
#         from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
#
#     # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
#     # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
#     max_frames = MAX_IMAGE_LENGTH  # 100
#
#     # The number of frames retained per second in the video.
#     video_framerate = 1
#
#     # Sampling Parameter
#     temperature = 0.01
#     top_p = None
#     num_beams = 1
#
#     disable_torch_init()
#     model_path = os.path.expanduser(model_path)
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, model_base, model_name, args.model_type
#     )
#
#     # # +++ 新增量化代码 +++
#     # if not args.disable_quant:
#     #     from quant_utils import VitaQuantizer
#     #
#     #     print("Applying GPTQ Quantization...")
#     #     quantizer = VitaQuantizer(bits=4)
#     #     model = quantizer.apply_quantization(model)
#     #     torch.cuda.empty_cache()
#     #     print("Quantization Completed")
#     # # +++ 结束新增 +++
#
#     model.resize_token_embeddings(len(tokenizer))
#
#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         vision_tower.load_model(path="/home/zqin3/vit/VITA/model_weights/InterViT-300M-448px")
#     image_processor = vision_tower.image_processor
#
#     audio_encoder = model.get_audio_encoder()
#     audio_encoder.to(dtype=torch.float16)
#     audio_processor = audio_encoder.audio_processor
#
#     model.eval()
#     if audio_path is not None:
#         audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
#         audio_length = audio.shape[0]
#         audio = torch.unsqueeze(audio, dim=0)
#         audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
#         audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
#         audios = dict()
#         audios["audios"] = audio.half().cuda()
#         audios["lengths"] = audio_length.half().cuda()
#         audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
#     else:
#         audio = torch.zeros(400, 80)
#         audio_length = audio.shape[0]
#         audio_for_llm_lens = 60
#         audio = torch.unsqueeze(audio, dim=0)
#         audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
#         audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
#         audios = dict()
#         audios["audios"] = audio.half().cuda()
#         audios["lengths"] = audio_length.half().cuda()
#         audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
#         # audios = None
#
#     # Check if the video exists
#     if video_path is not None:
#         video_frames, slice_len = _get_rawvideo_dec(
#             video_path,
#             image_processor,
#             max_frames=max_frames,
#             video_framerate=video_framerate,
#             image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
#         )
#         image_tensor = video_frames.half().cuda()
#         if audio_path:
#             qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
#         else:
#             qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
#         modality = "video"
#     elif image_path is not None:
#         # 加载图像并转换为 RGB
#         image = Image.open(image_path).convert("RGB")
#
#         # 缩放图像 以匹配模型输入大小
#         # 分割图像或选择多个 Patch
#         # 确保图像格式标准化
#         # 减少计算量，提高推理效率
#         if args.frameCat:
#             image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, image_size=448, use_thumbnail=True, img_mean=image_processor.image_mean)
#         else:
#             image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
#         assert len(p_num) == 1
#         image_tensor = model.process_images(image, model.config).to(
#             dtype=model.dtype, device="cuda"
#         )
#         if audio_path:
#             qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
#         else:
#             qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
#         modality = "image"
#     else:
#         image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
#         if audio_path:
#             qs = qs + DEFAULT_AUDIO_TOKEN
#         modality = "lang"
#
#     conv = conv_templates[conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt(modality)
#
#     if audio_path:
#         input_ids = (
#             tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#     else:
#         # 将 prompt（文本）转换为 Token ID
#         # 将 Token 组织成 PyTorch 张量（Tensor）
#         # 增加 Batch 维度
#         # 将张量移动到 GPU（CUDA）上进行计算
#         input_ids = (
#             tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#
#     start_time = time.time()
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             audios=audios,
#             do_sample=False,
#             temperature=temperature,
#             top_p=top_p,
#             num_beams=num_beams,
#             output_scores=True,
#             return_dict_in_generate=True,
#             max_new_tokens=1024,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria],
#             shared_v_pid_stride=None#2#16#8#4#1#None,
#         )
#     infer_time = time.time() - start_time
#     output_ids = output_ids.sequences
#     input_token_len = input_ids.shape[1]
#     if args.model_type == "mixtral-8x7b":
#         n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
#         if n_diff_input_output > 0:
#             print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
#             output_ids = output_ids[:, input_token_len:]
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
#
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[: -len(stop_str)]
#     outputs = outputs.strip()
#     print(outputs)
#     print(f"Time consume: {infer_time}")
#
#


# quant
# import argparse
# import os
# import time
# import sys
#
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
#
# # decord for video
# from decord import VideoReader, cpu
#
# # ============================================
# # ========== 1. GPTQ 相关类与函数 ============
# # ============================================
#
# DEBUG = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
#
# class Quantizer:
#     """
#     简化版量化器示例.
#     """
#     def __init__(self):
#         self.scale = None
#         self.zero = None
#         self.maxq = None
#         self.ready_flag = False
#
#     def configure(self, wbits=8, perchannel=True, sym=False, mse=False, trits=False):
#         """
#         设定量化参数(如4bit, 是否perchannel, 是否对称量化等).
#         """
#         self.wbits = wbits
#         self.perchannel = perchannel
#         self.sym = sym
#         self.mse = mse
#         self.trits = trits
#         self.maxq = 3**wbits - 1 if trits else 2**wbits - 1
#         self.ready_flag = False
#
#     def find_params(self, x, weight=True):
#         """
#         简化: 假设 x 的分布在 [x.min(), x.max()] 范围, 计算 scale/zero.
#         """
#         # 真实实现中需要考虑对称 / per-channel / mse 等情况
#         self.scale = (x.max() - x.min()) / self.maxq
#         self.zero = x.min()
#         self.ready_flag = True
#
#     def ready(self):
#         return self.ready_flag
#
#
# def quantize(x, scale, zero, maxq):
#     """
#     最基础的量化：((x - zero)/scale) -> clamp + round -> dequant.
#     """
#     q = (x - zero) / scale
#     q = torch.clamp(q, 0, maxq)
#     q = torch.round(q)
#     return q * scale + zero
#
#
# class GPTQ:
#     """
#     简化版 GPTQ 类: 收集 Hessian + 调用 Cholesky 分解 + 分块量化.
#     """
#     def __init__(self, layer):
#         self.layer = layer
#         self.dev = layer.weight.device
#         W = layer.weight.data.clone()
#         if isinstance(layer, nn.Conv2d):
#             W = W.flatten(1)
#         self.rows = W.shape[0]
#         self.columns = W.shape[1]
#         self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float16)
#         self.nsamples = 0
#         self.quantizer = None
#
#     def add_batch(self, inp, out):
#         """
#         收集 (inp * inp^T) 近似 Hessian.
#         """
#         if DEBUG:
#             self.inp1 = inp
#             self.out1 = out
#         if len(inp.shape) == 2:
#             inp = inp.unsqueeze(0)
#         tmp = inp.shape[0]
#
#         # 如果是线性层, 需要转置
#         if isinstance(self.layer, nn.Linear):
#             if len(inp.shape) == 3:
#                 inp = inp.reshape((-1, inp.shape[-1]))
#             inp = inp.t()
#         elif isinstance(self.layer, nn.Conv2d):
#             unfold = nn.Unfold(
#                 self.layer.kernel_size,
#                 dilation=self.layer.dilation,
#                 padding=self.layer.padding,
#                 stride=self.layer.stride
#             )
#             inp = unfold(inp)
#             inp = inp.permute([1, 0, 2])
#             inp = inp.flatten(1)
#
#         # 归一化 Hessian
#         self.H *= self.nsamples / (self.nsamples + tmp)
#         self.nsamples += tmp
#         inp = (inp.float() * (np.sqrt(2 / self.nsamples)))
#         self.H += inp.matmul(inp.t())
#
#     def fasterquant(self, blocksize=128, percdamp= 0.1, groupsize=-1, actorder=False, static_groups=False):
#         """
#         GPTQ 核心: 使用 Hessian self.H + 强力对角线加大(Damp) + Cholesky 分解，分块对 self.layer.weight 做最优量化。
#         如果碰到 "input not positive-definite" 或 OOM，可再加大 lam。
#         """
#
#         # 1) 先复制/转换当前层的权重
#         W = self.layer.weight.data.clone()
#         if isinstance(self.layer, nn.Conv2d):
#             W = W.flatten(1)
#         W = W.float()
#
#         # 若量化器(quantizer)还未 find_params，就先基于W分析量化范围
#         if not self.quantizer.ready():
#             self.quantizer.find_params(W, weight=True)
#
#         # 2) 取出 Hessian 矩阵 H，并释放 self.H
#         H = self.H
#         del self.H
#
#         # 对角线为0的列(无梯度)记为dead
#         dead = torch.diag(H) == 0
#         H[dead, dead] = 1
#         W[:, dead] = 0
#
#         # 3) (可选) actorder：根据 H 的对角线大小排序
#         if actorder:
#             perm = torch.argsort(torch.diag(H), descending=True)
#             W = W[:, perm]
#             H = H[perm][:, perm]
#             invperm = torch.argsort(perm)
#         else:
#             perm = None
#             invperm = None
#
#         # 量化后权重/损失 的缓存
#         Q = torch.zeros_like(W)
#         Losses = torch.zeros_like(W)
#
#         # 4) 强力 Damp：在对角线上加大若干倍 (lam)
#         #    percdamp=1 表示默认阻尼力度，
#         #    这里 lam=10.0 是再加 10 倍，可根据情况加大
#         damp = percdamp * torch.mean(torch.diag(H))
#         diag = torch.arange(self.columns, device=self.dev)
#
#         lam = 10.0
#         H[diag, diag] += lam * damp
#
#         # 转 float32 以支持 Cholesky
#         H = H.to(torch.float32)
#
#         # (可选) 对称化，减少数值误差
#         H = 0.5 * (H + H.transpose(-1, -2))
#
#         # 5) 试图做 Cholesky
#         try:
#             H = torch.linalg.cholesky(H)
#         except RuntimeError:
#             print("[GPTQ] Cholesky still fails, skip this layer or try bigger lam/damp?")
#             # 如果想跳过这个层，可以加 return；或可再+ debug
#             return
#
#         # 6) 做逆 / 再次 Cholesky
#         # H = torch.cholesky_inverse(H)
#         #
#         # H = torch.linalg.cholesky(H, upper=True)
#         Hinv = H
#
#         # 7) 分块处理(一次处理 blocksize 列)，进行最优量化
#         for i1 in range(0, self.columns, blocksize):
#             i2 = min(i1 + blocksize, self.columns)
#             W1 = W[:, i1:i2].clone()
#             Hinv1 = Hinv[i1:i2, i1:i2]
#
#             Err1 = torch.zeros_like(W1)
#             Q1 = torch.zeros_like(W1)
#             Losses1 = torch.zeros_like(W1)
#
#             count = i2 - i1
#             for i in range(count):
#                 w = W1[:, i]
#                 d = Hinv1[i, i]
#
#                 # 量化
#                 q = quantize(
#                     w.unsqueeze(1),
#                     self.quantizer.scale,
#                     self.quantizer.zero,
#                     self.quantizer.maxq
#                 ).flatten()
#
#                 Q1[:, i] = q
#                 Losses1[:, i] = (w - q) ** 2 / (d ** 2)
#
#                 # 误差补偿
#                 err1 = (w - q) / d
#                 W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
#                 Err1[:, i] = err1
#
#             Q[:, i1:i2] = Q1
#             Losses[:, i1:i2] = Losses1 / 2  # 损失
#
#             # 更新剩余列
#             W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
#
#         # 8) 打印量化损失
#         err_total = torch.sum(Losses).item()
#         print(f"[GPTQ] fasterquant finish, error = {err_total:.4f}")
#
#         # 如果 actorder 开启，需要把量化后 Q 再排回原顺序
#         if actorder and invperm is not None:
#             Q = Q[:, invperm]
#
#         # 9) 恢复卷积层形状
#         if isinstance(self.layer, nn.Conv2d):
#             Q = Q.reshape(self.layer.weight.shape)
#
#         # 赋值回原层
#         self.layer.weight.data = Q.to(self.layer.weight.dtype)
#         torch.cuda.synchronize()
#
#     def free(self):
#         self.H = None
#         torch.cuda.empty_cache()
#
#
# def find_layers(module, layers=(nn.Linear,), prefix=''):
#     """
#     递归遍历 module, 收集给定类型(layers)的子模块.
#     返回 dict: { 'module_full_name': module_object }
#     """
#     res = {}
#     for name, mod in module.named_children():
#         full_name = f"{prefix}.{name}" if prefix else name
#         if isinstance(mod, layers):
#             res[full_name] = mod
#         else:
#             res.update(find_layers(mod, layers, full_name))
#     return res
#
#
# def vita_sequential(model, calibration_loader, device='cuda'):
#     model.to(device)
#     model.eval()
#
#     # 1. 找出需要量化的 nn.Linear 层
#     layers_dict = find_layers(model, layers=(nn.Linear,))
#     gptq_objs = {}
#     for name, layer in layers_dict.items():
#         g = GPTQ(layer)
#         g.quantizer = Quantizer()
#         # 配置量化参数(4bit, perchannel等)
#         g.quantizer.configure(wbits=4, perchannel=True, sym=False, mse=False)
#         gptq_objs[name] = g
#
#     # 2. 注册 Hook，收集 Hessian
#     def make_hook(n):
#         def tmp_hook(mod, inp, out):
#             gptq_objs[n].add_batch(inp[0], out)
#         return tmp_hook
#
#     handles = []
#     for n, lyr in layers_dict.items():
#         h = lyr.register_forward_hook(make_hook(n))
#         handles.append(h)
#
#     # 3. 在校准阶段，只把文本给模型
#     with torch.no_grad():
#         for batch in calibration_loader:
#             # 这里 batch["input_ids"] 形状应为 (1, 10)
#             batch["input_ids"] = batch["input_ids"].squeeze(1)
#             print("calibration input_ids shape =", batch["input_ids"].shape)
#             # 注意只给 "input_ids"
#             model(**{"input_ids": batch["input_ids"]})
#
#     # 4. 移除 Hook 并调用 fasterquant
#     for h in handles:
#         h.remove()
#
#     for n in layers_dict:
#         print(f"[GPTQ] start quant layer: {n}")
#         gptq_objs[n].fasterquant()
#         gptq_objs[n].free()
#
#     return model
#
#
#
# def get_calibration_dataloader(tokenizer, device='cuda'):
#     """
#     简单示例: 返回一个只含文本 input_ids 的 DataLoader，
#     确保形状是 (batch=1, seq_len=10).
#     """
#     from torch.utils.data import Dataset, DataLoader
#
#     class FakeCalibDataset(Dataset):
#         def __len__(self):
#             return 2  # 几条校准数据都行
#         def __getitem__(self, idx):
#             # 这里仅返回纯文本 input_ids
#             data = {}
#             # 保证是 (1, 10) => batch=1, seq_len=10
#             data["input_ids"] = torch.zeros(
#                 (1, 10),
#                 dtype=torch.long,
#                 device=device
#             )
#             # 不返回 images / audios
#             return data
#
#     ds = FakeCalibDataset()
#     dl = DataLoader(ds, batch_size=1, shuffle=False)
#     return dl
#
#
#
# # ============================================
# # ========== 2. VITA 原有推理逻辑 ============
# # ============================================
# from vita.constants import (
#     DEFAULT_AUDIO_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_VIDEO_TOKEN,
#     IGNORE_INDEX,
#     IMAGE_TOKEN_INDEX,
#     MAX_IMAGE_LENGTH,
# )
# from vita.conversation import SeparatorStyle, conv_templates
# from vita.model.builder import load_pretrained_model
# from vita.util.mm_utils import (
#     KeywordsStoppingCriteria,
#     get_model_name_from_path,
#     tokenizer_image_audio_token,
#     tokenizer_image_token,
# )
# from vita.util.utils import disable_torch_init
#
# def _get_rawvideo_dec(
#     video_path,
#     image_processor,
#     max_frames=MAX_IMAGE_LENGTH,
#     min_frames=4,
#     image_resolution=384,
#     video_framerate=1,
#     s=None,
#     e=None,
#     image_aspect_ratio="pad",
# ):
#     # 原有的视频解码函数(保持不变)
#     from decord import VideoReader, cpu
#
#     if s is None:
#         start_time, end_time = None, None
#     else:
#         start_time = int(s)
#         end_time = int(e)
#         start_time = start_time if start_time >= 0.0 else 0.0
#         end_time = end_time if end_time >= 0.0 else 0.0
#         if start_time > end_time:
#             start_time, end_time = end_time, start_time
#         elif start_time == end_time:
#             end_time = start_time + 1
#
#     if os.path.exists(video_path):
#         vreader = VideoReader(video_path, ctx=cpu(0))
#     else:
#         print(video_path)
#         raise FileNotFoundError
#
#     fps = vreader.get_avg_fps()
#     f_start = 0 if start_time is None else int(start_time * fps)
#     f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
#     num_frames = f_end - f_start + 1
#     if num_frames > 0:
#         sample_fps = int(video_framerate)
#         t_stride = int(round(float(fps) / sample_fps))
#         all_pos = list(range(f_start, f_end + 1, t_stride))
#         if len(all_pos) > max_frames:
#             import numpy as np
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
#             ]
#         elif len(all_pos) < min_frames:
#             import numpy as np
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
#             ]
#         else:
#             sample_pos = all_pos
#
#         patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
#         if image_aspect_ratio == "pad":
#             def expand2square(pil_img, background_color):
#                 width, height = pil_img.size
#                 if width == height:
#                     return pil_img
#                 elif width > height:
#                     from PIL import Image
#                     result = Image.new(pil_img.mode, (width, width), background_color)
#                     result.paste(pil_img, (0, (width - height)//2))
#                     return result
#                 else:
#                     from PIL import Image
#                     result = Image.new(pil_img.mode, (height, height), background_color)
#                     result.paste(pil_img, ((height - width)//2, 0))
#                     return result
#
#             patch_images = [
#                 expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
#                 for i in patch_images
#             ]
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#         else:
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#         patch_images = torch.stack(patch_images)
#         slice_len = patch_images.shape[0]
#         return patch_images, slice_len
#     else:
#         print("video path: {} error.".format(video_path))
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process model and video paths.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
#     parser.add_argument("--model_base", type=str, default=None)
#     parser.add_argument("--video_path", type=str, default=None)
#     parser.add_argument("--image_path", type=str, default=None)
#     parser.add_argument("--audio_path", type=str, default=None)
#     parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
#     parser.add_argument("--conv_mode", type=str, default="mixtral_two")
#     parser.add_argument("--question", type=str, default="")
#     parser.add_argument("--frameCat", action='store_true')
#     parser.add_argument("--disable_quant", type=int, default=0, help="0=enable GPTQ, 1=disable")
#
#     args = parser.parse_args()
#
#     model_path = os.path.expanduser(args.model_path)
#     model_name = get_model_name_from_path(model_path)
#     disable_torch_init()
#
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, args.model_base, model_name, args.model_type
#     )
#
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # (新增) 若 disable_quant=0, 执行 GPTQ 量化
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     if args.disable_quant == 0:
#         print("[GPTQ] Collecting calibration data & applying GPTQ quant...")
#         # 1) 构造 calibration_loader
#         calibration_loader = get_calibration_dataloader(tokenizer, device='cuda')
#         # 2) 调用 vita_sequential
#         model = vita_sequential(model, calibration_loader, device='cuda')
#         torch.cuda.empty_cache()
#         print("[GPTQ] Done quantization.\n")
#
#         # 追加一行，保存模型权重
#         torch.save(model.state_dict(), "vita_quantized.pth")
#         print("Quantized model saved to vita_quantized.pth")
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#     model.resize_token_embeddings(len(tokenizer))
#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         vision_tower.load_model(path="/home/.../InterViT-300M-448px")
#     image_processor = vision_tower.image_processor
#
#     audio_encoder = model.get_audio_encoder()
#     audio_encoder.to(dtype=torch.float16)
#     audio_processor = audio_encoder.audio_processor
#
#     model.eval()
#
#     # 下面是跟原先一样的音视频处理:
#     video_path = args.video_path
#     image_path = args.image_path
#     audio_path = args.audio_path
#     qs = args.question
#     assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
#     conv_mode = args.conv_mode
#
#     if args.frameCat:
#         from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
#     else:
#         from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
#
#     max_frames = MAX_IMAGE_LENGTH
#     video_framerate = 1
#
#     temperature = 0.01
#     top_p = None
#     num_beams = 1
#
#     if audio_path is not None:
#         audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
#         audio_length = audio.shape[0]
#         audio = audio.unsqueeze(0)
#         audio_length = torch.tensor(audio_length).unsqueeze(0)
#         audio_for_llm_lens = torch.tensor(audio_for_llm_lens).unsqueeze(0)
#         audios = {
#             "audios": audio.half().cuda(),
#             "lengths": audio_length.half().cuda(),
#             "lengths_for_llm": audio_for_llm_lens.cuda()
#         }
#     else:
#         audio = torch.zeros(400, 80)
#         audio_length = audio.shape[0]
#         audio_for_llm_lens = 60
#         audio = audio.unsqueeze(0)
#         audio_length = torch.tensor(audio_length).unsqueeze(0)
#         audio_for_llm_lens = torch.tensor(audio_for_llm_lens).unsqueeze(0)
#         audios = {
#             "audios": audio.half().cuda(),
#             "lengths": audio_length.half().cuda(),
#             "lengths_for_llm": audio_for_llm_lens.cuda()
#         }
#
#     if video_path is not None:
#         video_frames, slice_len = _get_rawvideo_dec(
#             video_path,
#             image_processor,
#             max_frames=max_frames,
#             video_framerate=video_framerate,
#             image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
#         )
#         image_tensor = video_frames.half().cuda()
#         from vita.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
#         if audio_path:
#             qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
#         else:
#             qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
#         modality = "video"
#     elif image_path is not None:
#         image = Image.open(image_path).convert("RGB")
#         if args.frameCat:
#             image, p_num = dynamic_preprocess(
#                 image,
#                 min_num=2,
#                 max_num=12,
#                 image_size=448,
#                 use_thumbnail=True,
#                 img_mean=image_processor.image_mean
#             )
#         else:
#             image, p_num = dynamic_preprocess(
#                 image,
#                 min_num=1,
#                 max_num=12,
#                 image_size=448,
#                 use_thumbnail=True
#             )
#         assert len(p_num) == 1
#         image_tensor = model.process_images(image, model.config).to(
#             dtype=model.dtype, device="cuda"
#         )
#         from vita.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
#         if audio_path:
#             qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
#         else:
#             qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
#         modality = "image"
#     else:
#         image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
#         if audio_path:
#             qs = qs + DEFAULT_AUDIO_TOKEN
#         modality = "lang"
#
#     from vita.conversation import SeparatorStyle, conv_templates
#     conv = conv_templates[conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt(modality)
#
#     from vita.util.mm_utils import (
#         KeywordsStoppingCriteria,
#         tokenizer_image_audio_token,
#         tokenizer_image_token,
#         IMAGE_TOKEN_INDEX
#     )
#
#     if audio_path:
#         input_ids = (
#             tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#     else:
#         input_ids = (
#             tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#
#     start_time = time.time()
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             audios=audios,
#             do_sample=False,
#             temperature=temperature,
#             top_p=top_p,
#             num_beams=num_beams,
#             output_scores=True,
#             return_dict_in_generate=True,
#             max_new_tokens=1024,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria],
#         )
#     infer_time = time.time() - start_time
#     output_ids = output_ids.sequences
#     input_token_len = input_ids.shape[1]
#
#     if args.model_type == "mixtral-8x7b":
#         n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
#         if n_diff_input_output > 0:
#             print(f"[Warning] {n_diff_input_output} output_ids not the same as input_ids")
#             output_ids = output_ids[:, input_token_len:]
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[: -len(stop_str)]
#     outputs = outputs.strip()
#
#     print(outputs)
#     print(f"Time consume: {infer_time:.2f} s")


#!/usr/bin/env python3
# coding: utf-8

# import argparse
# import os
# import time
#
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# from decord import VideoReader, cpu
#
# # ========== VITA 相关 import ==========
# from vita.constants import (
#     DEFAULT_AUDIO_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_VIDEO_TOKEN,
#     IGNORE_INDEX,
#     IMAGE_TOKEN_INDEX,
#     MAX_IMAGE_LENGTH,
# )
# from vita.conversation import SeparatorStyle, conv_templates
# from vita.model.builder import load_pretrained_model
# from vita.util.mm_utils import (
#     KeywordsStoppingCriteria,
#     get_model_name_from_path,
#     tokenizer_image_audio_token,
#     tokenizer_image_token,
# )
# from vita.util.utils import disable_torch_init
#
# # ========== GPTQ 相关类与函数 ==========
# DEBUG = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
#
# class Quantizer:
#     """ 简化版量化器示例. """
#     def __init__(self):
#         self.scale = None
#         self.zero = None
#         self.maxq = None
#         self.ready_flag = False
#
#     def configure(self, wbits=4, perchannel=True, sym=False, mse=False, trits=False):
#         """
#         设定量化参数(如4bit, 是否perchannel, 是否对称量化等).
#         """
#         self.wbits = wbits
#         self.perchannel = perchannel
#         self.sym = sym
#         self.mse = mse
#         self.trits = trits
#         self.maxq = (3**wbits - 1) if trits else (2**wbits - 1)
#         self.ready_flag = False
#
#     def find_params(self, x, weight=True):
#         """
#         简化: 假设 x 的分布在 [x.min(), x.max()] 范围, 计算 scale/zero.
#         """
#         self.scale = (x.max() - x.min()) / self.maxq
#         self.zero = x.min()
#         self.ready_flag = True
#
#     def ready(self):
#         return self.ready_flag
#
#
# def quantize(x, scale, zero, maxq):
#     """
#     最基础的量化：((x - zero)/scale) -> clamp + round -> dequant.
#     """
#     q = (x - zero) / scale
#     q = torch.clamp(q, 0, maxq)
#     q = torch.round(q)
#     return q * scale + zero
#
#
# class GPTQ:
#     """
#     简化版 GPTQ: 收集 Hessian + Cholesky + 分块量化
#     """
#     def __init__(self, layer):
#         self.layer = layer
#         self.dev = layer.weight.device
#         W = layer.weight.data.clone()
#         if isinstance(layer, nn.Conv2d):
#             W = W.flatten(1)
#         self.rows = W.shape[0]
#         self.columns = W.shape[1]
#         self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float16)
#         self.nsamples = 0
#         self.quantizer = None
#
#     def add_batch(self, inp, out):
#         """ 收集 (inp * inp^T) 近似 Hessian. """
#         if DEBUG:
#             self.inp1 = inp
#             self.out1 = out
#         if len(inp.shape) == 2:
#             inp = inp.unsqueeze(0)
#         tmp = inp.shape[0]
#
#         # 如果是 linear 层，需要转置
#         if isinstance(self.layer, nn.Linear):
#             if len(inp.shape) == 3:
#                 inp = inp.reshape((-1, inp.shape[-1]))
#             inp = inp.t()
#         elif isinstance(self.layer, nn.Conv2d):
#             unfold = nn.Unfold(
#                 self.layer.kernel_size,
#                 dilation=self.layer.dilation,
#                 padding=self.layer.padding,
#                 stride=self.layer.stride
#             )
#             inp = unfold(inp)
#             inp = inp.permute([1, 0, 2])
#             inp = inp.flatten(1)
#
#         # 归一化 Hessian
#         self.H *= self.nsamples / (self.nsamples + tmp)
#         self.nsamples += tmp
#         inp = inp.float() * (np.sqrt(2 / self.nsamples))
#         self.H += inp.matmul(inp.t())
#
#     def fasterquant(self, blocksize=128, percdamp=0.1, groupsize=-1, actorder=False, static_groups=False):
#         """
#         使用 Hessian self.H + 强力 Damp + Cholesky 分解 + 分块，对 self.layer.weight 做最优量化。
#         """
#         W = self.layer.weight.data.clone()
#         if isinstance(self.layer, nn.Conv2d):
#             W = W.flatten(1)
#         W = W.float()
#
#         if not self.quantizer.ready():
#             self.quantizer.find_params(W, weight=True)
#
#         H = self.H
#         del self.H
#         dead = torch.diag(H) == 0
#         H[dead, dead] = 1
#         W[:, dead] = 0
#
#         if actorder:
#             perm = torch.argsort(torch.diag(H), descending=True)
#             W = W[:, perm]
#             H = H[perm][:, perm]
#             invperm = torch.argsort(perm)
#         else:
#             perm = None
#             invperm = None
#
#         Q = torch.zeros_like(W)
#         Losses = torch.zeros_like(W)
#
#         damp = percdamp * torch.mean(torch.diag(H))
#         diag = torch.arange(self.columns, device=self.dev)
#
#         lam = 10.0  # 强力 *10
#         H[diag, diag] += lam * damp
#
#         H = H.to(torch.float32)
#         # 对称化
#         H = 0.5 * (H + H.transpose(-1, -2))
#
#         # Cholesky
#         try:
#             H = torch.linalg.cholesky(H)
#         except RuntimeError:
#             print("[GPTQ] Cholesky fails, skip this layer or try bigger lam/damp.")
#             return
#
#         Hinv = H
#
#         # 分块
#         for i1 in range(0, self.columns, blocksize):
#             i2 = min(i1 + blocksize, self.columns)
#             W1 = W[:, i1:i2].clone()
#             Hinv1 = Hinv[i1:i2, i1:i2]
#
#             Err1 = torch.zeros_like(W1)
#             Q1 = torch.zeros_like(W1)
#             Losses1 = torch.zeros_like(W1)
#
#             count = i2 - i1
#             for i in range(count):
#                 w = W1[:, i]
#                 d = Hinv1[i, i]
#                 # 量化
#                 q = quantize(w.unsqueeze(1), self.quantizer.scale,
#                              self.quantizer.zero, self.quantizer.maxq).flatten()
#                 Q1[:, i] = q
#                 Losses1[:, i] = (w - q)**2 / d**2
#
#                 err1 = (w - q)/d
#                 W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
#                 Err1[:, i] = err1
#
#             Q[:, i1:i2] = Q1
#             Losses[:, i1:i2] = Losses1/2
#
#             W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
#
#         err_total = torch.sum(Losses).item()
#         print(f"[GPTQ] fasterquant finish, error = {err_total:.4f}")
#
#         if actorder and invperm is not None:
#             Q = Q[:, invperm]
#
#         if isinstance(self.layer, nn.Conv2d):
#             Q = Q.reshape(self.layer.weight.shape)
#
#         self.layer.weight.data = Q.to(self.layer.weight.dtype)
#         torch.cuda.synchronize()
#
#     def free(self):
#         self.H = None
#         torch.cuda.empty_cache()
#
#
# def find_layers(module, layers=(nn.Linear,), prefix=''):
#     """ 递归遍历 module, 收集指定类型(layers)的子模块 """
#     res = {}
#     for name, mod in module.named_children():
#         full_name = f"{prefix}.{name}" if prefix else name
#         if isinstance(mod, layers):
#             res[full_name] = mod
#         else:
#             res.update(find_layers(mod, layers, full_name))
#     return res
#
#
# def vita_sequential(model, calibration_loader, device='cuda'):
#     model.to(device)
#     model.eval()
#
#     layers_dict = find_layers(model, layers=(nn.Linear,))
#     gptq_objs = {}
#     for name, layer in layers_dict.items():
#         g = GPTQ(layer)
#         g.quantizer = Quantizer()
#         g.quantizer.configure(wbits=4, perchannel=True, sym=False, mse=False)
#         gptq_objs[name] = g
#
#     def make_hook(n):
#         def tmp_hook(mod, inp, out):
#             gptq_objs[n].add_batch(inp[0], out)
#         return tmp_hook
#
#     handles = []
#     for n, lyr in layers_dict.items():
#         h = lyr.register_forward_hook(make_hook(n))
#         handles.append(h)
#
#     with torch.no_grad():
#         for batch in calibration_loader:
#             # 这里 batch["input_ids"] 形状应为 (1, 10)
#             batch["input_ids"] = batch["input_ids"].squeeze(1)
#             print("calibration input_ids shape =", batch["input_ids"].shape)
#             model(**{"input_ids": batch["input_ids"]})
#
#     for h in handles:
#         h.remove()
#
#     for n in layers_dict:
#         print(f"[GPTQ] start quant layer: {n}")
#         gptq_objs[n].fasterquant()
#         gptq_objs[n].free()
#
#     return model
#
# def get_calibration_dataloader(device='cuda'):
#     """
#     简单示例: 返回一个只含文本 input_ids 的 DataLoader，
#     确保形状是 (1,10). 仅作 GPTQ 校准之用.
#     """
#     from torch.utils.data import Dataset, DataLoader
#     class FakeCalibDataset(Dataset):
#         def __len__(self):
#             return 2
#         def __getitem__(self, idx):
#             data = {}
#             data["input_ids"] = torch.zeros((1,10), dtype=torch.long, device=device)
#             return data
#     ds = FakeCalibDataset()
#     dl = DataLoader(ds, batch_size=1, shuffle=False)
#     return dl
#
# # ========== 读取视频的函数(保持不变) ==========
# def _get_rawvideo_dec(
#     video_path,
#     image_processor,
#     max_frames=MAX_IMAGE_LENGTH,
#     min_frames=4,
#     image_resolution=384,
#     video_framerate=1,
#     s=None,
#     e=None,
#     image_aspect_ratio="pad",
# ):
#     from decord import VideoReader, cpu
#     import numpy as np
#     if s is None:
#         start_time, end_time = None, None
#     else:
#         start_time = int(s)
#         end_time = int(e)
#         start_time = max(start_time, 0)
#         end_time = max(end_time, 0)
#         if start_time> end_time:
#             start_time, end_time = end_time, start_time
#         elif start_time==end_time:
#             end_time = start_time+1
#
#     if os.path.exists(video_path):
#         vreader = VideoReader(video_path, ctx=cpu(0))
#     else:
#         print(video_path)
#         raise FileNotFoundError
#
#     fps = vreader.get_avg_fps()
#     f_start = 0 if start_time is None else int(start_time*fps)
#     f_end = int(min(1000000000 if end_time is None else end_time*fps, len(vreader)-1))
#     num_frames = f_end - f_start+1
#     if num_frames>0:
#         sample_fps = int(video_framerate)
#         t_stride = int(round(float(fps)/sample_fps))
#         all_pos = list(range(f_start,f_end+1,t_stride))
#         if len(all_pos)>max_frames:
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos)-1, num=max_frames, dtype=int)
#             ]
#         elif len(all_pos)<min_frames:
#             sample_pos = [
#                 all_pos[_] for _ in np.linspace(0, len(all_pos)-1, num=min_frames, dtype=int)
#             ]
#         else:
#             sample_pos = all_pos
#         patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
#         if image_aspect_ratio=="pad":
#             def expand2square(pil_img, background_color):
#                 w,h= pil_img.size
#                 if w==h:
#                     return pil_img
#                 elif w>h:
#                     from PIL import Image
#                     res = Image.new(pil_img.mode,(w,w), background_color)
#                     res.paste(pil_img,(0,(w-h)//2))
#                     return res
#                 else:
#                     from PIL import Image
#                     res = Image.new(pil_img.mode,(h,h), background_color)
#                     res.paste(pil_img,((h-w)//2,0))
#                     return res
#             patch_images = [
#                 expand2square(i, tuple(int(x*255) for x in image_processor.image_mean))
#                 for i in patch_images
#             ]
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#         else:
#             patch_images = [
#                 image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
#                 for i in patch_images
#             ]
#         patch_images = torch.stack(patch_images)
#         slice_len = patch_images.shape[0]
#         return patch_images, slice_len
#     else:
#         print("video path: {} error.".format(video_path))
#         return None,None
#
# def main():
#     parser = argparse.ArgumentParser(description="Multi-Modal + GPTQ + benchmark in a single script")
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--model_base", type=str, default=None)
#     parser.add_argument("--video_path", type=str, default=None)
#     parser.add_argument("--image_path", type=str, default=None)
#     parser.add_argument("--audio_path", type=str, default=None)
#     parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
#     parser.add_argument("--conv_mode", type=str, default="mixtral_two")
#     parser.add_argument("--question", type=str, default="")
#     parser.add_argument("--frameCat", action='store_true')
#     parser.add_argument("--disable_quant", type=int, default=0,
#                         help="Whether to skip GPTQ quant. 0=enable GPTQ,1=skip.")
#     # 新增：循环推理次数
#     parser.add_argument("--bench_rounds", type=int, default=100,
#                         help="Number of repeated forward calls to measure average inference time.")
#
#     args= parser.parse_args()
#
#     disable_torch_init()
#     model_path = os.path.expanduser(args.model_path)
#     model_name = get_model_name_from_path(model_path)
#     from vita.model.builder import load_pretrained_model
#
#     # 1) 加载原模型
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, args.model_base, model_name, args.model_type
#     )
#     model.eval().cuda()
#
#     # 2) 如果 disable_quant=0, 做 GPTQ
#     if args.disable_quant==0:
#         print("[GPTQ] Collecting calibration data & applying GPTQ quant...")
#         calib_loader = get_calibration_dataloader(device="cuda")
#         model = vita_sequential(model, calib_loader, device="cuda")
#         torch.cuda.empty_cache()
#         print("[GPTQ] Done quantization.\n")
#         torch.save(model.state_dict(),"vita_quantized.pth")
#         print("Quantized model saved to vita_quantized.pth")
#     else:
#         # skip GPTQ,如果已有量化文件就载入
#         if os.path.exists("vita_quantized.pth"):
#             print("[GPTQ] Loading quantized weights from vita_quantized.pth ...")
#             sd = torch.load("vita_quantized.pth", map_location="cuda")
#             missing, unexpected = model.load_state_dict(sd, strict=False)
#             print(f"Missing keys: {missing}, Unexpected: {unexpected}")
#             print("[GPTQ] Loaded quant.\n")
#         else:
#             print("[WARN] no vita_quantized.pth found. using original weights.\n")
#
#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         vision_tower.load_model(path="/home/.../InterViT-300M-448px")
#     image_processor = vision_tower.image_processor
#
#     audio_encoder = model.get_audio_encoder()
#     audio_encoder.to(dtype=torch.float16)
#     audio_processor = audio_encoder.audio_processor
#
#     model.eval()
#
#     # 构造多模态输入
#     qs = args.question
#     if args.frameCat:
#         from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
#     else:
#         from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
#
#     video_path = args.video_path
#     image_path = args.image_path
#     audio_path = args.audio_path
#
#     assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
#
#     if audio_path is not None:
#         audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
#         audio_length = audio.shape[0]
#         audio = audio.unsqueeze(0)
#         audio_length = torch.tensor(audio_length).unsqueeze(0)
#         audio_for_llm_lens = torch.tensor(audio_for_llm_lens).unsqueeze(0)
#         audios = {
#             "audios": audio.half().cuda(),
#             "lengths": audio_length.half().cuda(),
#             "lengths_for_llm": audio_for_llm_lens.cuda()
#         }
#     else:
#         audio = torch.zeros((400,80))
#         audio_length = audio.shape[0]
#         audio_for_llm_lens = 60
#         audio = audio.unsqueeze(0)
#         audio_length = torch.tensor(audio_length).unsqueeze(0)
#         audio_for_llm_lens = torch.tensor(audio_for_llm_lens).unsqueeze(0)
#         audios = {
#             "audios": audio.half().cuda(),
#             "lengths": audio_length.half().cuda(),
#             "lengths_for_llm": audio_for_llm_lens.cuda()
#         }
#
#     from vita.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
#     if video_path is not None:
#         video_frames, slice_len = _get_rawvideo_dec(
#             video_path,
#             image_processor
#         )
#         image_tensor = video_frames.half().cuda()
#         if audio_path:
#             qs = (DEFAULT_IMAGE_TOKEN*slice_len)+"\n"+qs+DEFAULT_AUDIO_TOKEN
#         else:
#             qs = (DEFAULT_IMAGE_TOKEN*slice_len)+"\n"+qs
#         modality="video"
#     elif image_path is not None:
#         im = Image.open(image_path).convert("RGB")
#         if args.frameCat:
#             im, p_num = dynamic_preprocess(im, min_num=2,max_num=12,image_size=448,
#                                            use_thumbnail=True,img_mean=image_processor.image_mean)
#         else:
#             im, p_num = dynamic_preprocess(im, min_num=1,max_num=12,image_size=448,use_thumbnail=True)
#         assert len(p_num)==1
#         image_tensor = model.process_images(im, model.config).to(dtype=model.dtype,device="cuda")
#         if audio_path:
#             qs=(DEFAULT_IMAGE_TOKEN*p_num[0])+"\n"+qs+DEFAULT_AUDIO_TOKEN
#         else:
#             qs=(DEFAULT_IMAGE_TOKEN*p_num[0])+"\n"+qs
#         modality="image"
#     else:
#         image_tensor=torch.zeros((1,3,448,448),dtype=model.dtype,device="cuda")
#         if audio_path:
#             qs=qs+DEFAULT_AUDIO_TOKEN
#         modality="lang"
#
#     conv_mode=args.conv_mode
#     from vita.conversation import SeparatorStyle, conv_templates
#     conv = conv_templates[conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt=conv.get_prompt(modality)
#
#     from vita.util.mm_utils import (
#         KeywordsStoppingCriteria,
#         tokenizer_image_audio_token,
#         tokenizer_image_token,
#         IMAGE_TOKEN_INDEX
#     )
#
#     if audio_path:
#         input_ids=(
#             tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0).cuda()
#         )
#     else:
#         input_ids=(
#             tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0).cuda()
#         )
#
#     stop_str = conv.sep if conv.sep_style!=SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria=KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#
#     # =========== 这里循环 bench_rounds 次，统计平均耗时 + 峰值显存 =============
#     n_rounds = args.bench_rounds
#     times=[]
#     peak_mem=0.0
#
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#
#     final_output_ids=None
#
#     for i in range(n_rounds):
#         torch.cuda.reset_peak_memory_stats("cuda")
#
#         t0=time.time()
#         with torch.inference_mode():
#             out = model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 audios=audios,
#                 do_sample=False,
#                 temperature=0.01,
#                 top_p=None,
#                 num_beams=1,
#                 output_scores=True,
#                 return_dict_in_generate=True,
#                 max_new_tokens=1024,
#                 use_cache=True,
#                 stopping_criteria=[stopping_criteria],
#             )
#         torch.cuda.synchronize()
#         dt=time.time()-t0
#         times.append(dt)
#
#         final_output_ids = out.sequences
#         mem_now = torch.cuda.max_memory_allocated("cuda")/(1024**2)
#         if mem_now>peak_mem:
#             peak_mem=mem_now
#
#     avg_time=sum(times)/len(times)
#
#     # 拿最后一次结果
#     output_ids=final_output_ids
#     input_token_len= input_ids.shape[1]
#     if args.model_type=="mixtral-8x7b":
#         n_diff=(input_ids!=output_ids[:,:input_token_len]).sum().item()
#         if n_diff>0:
#             print(f"[Warning]{n_diff} output_ids not the same as input_ids")
#             output_ids=output_ids[:,input_token_len:]
#     from transformers import PreTrainedTokenizer
#     outputs= tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
#     outputs=outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs=outputs[:-len(stop_str)]
#     outputs=outputs.strip()
#
#     print("\n===== Benchmark Results =====")
#     print(f"Rounds: {n_rounds}")
#     print(f"Average time per round: {avg_time:.4f} s")
#     print(f"Peak GPU mem usage: {peak_mem:.1f} MB")
#
#     print("\n===== LLM Output (last round) =====")
#     print(outputs)
#
#
# if __name__=="__main__":
#     main()


#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from decord import VideoReader, cpu

# ========== VITA 相关 import ==========
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init

# =============== 你给出的新的量化逻辑 ===============
def new_quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero

    # 常规情况
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def new_find_params(self, x, weight=False):
    dev = x.device
    # 假设 self.maxq 是 int，这里转成 tensor
    self.maxq = torch.tensor(self.maxq, device=dev, dtype=torch.float32)

    shape = x.shape
    if self.perchannel:
        # flatten(1) or transpose
        if weight:
            x = x.flatten(1)
        else:
            if len(shape) == 4:
                x = x.permute([1, 0, 2, 3])
                x = x.flatten(1)
            elif len(shape) == 3:
                x = x.reshape((-1, shape[-1])).t()
            elif len(shape) == 2:
                x = x.t()
    else:
        x = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(dim=1)[0], tmp)
    xmax = torch.maximum(x.max(dim=1)[0], tmp)

    # 对称量化
    if self.sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp_mask = xmin < 0
        if torch.any(tmp_mask):
            xmin[tmp_mask] = -xmax[tmp_mask]

    zero_mask = (xmin == 0) & (xmax == 0)
    xmin[zero_mask] = -1.0
    xmax[zero_mask] = +1.0

    if self.maxq.item() < 0:
        self.scale = xmax
        self.zero = xmin
    else:
        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq.item() + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

    # 如果有需要在此加 mse / grid / norm 等高级策略
    # ...

    self.ready_flag = True

    # perchannel 完成后还原 shape
    if weight:
        shape2 = [-1] + [1] * (len(shape)-1)
        self.scale = self.scale.reshape(shape2)
        self.zero  = self.zero.reshape(shape2)
    else:
        pass


# =============== 定义一个新的 Quantizer 类 ===============
class Quantizer:
    def __init__(self):
        self.scale = None
        self.zero = None
        self.maxq = None
        self.ready_flag = False
        self.perchannel = False
        self.sym = False

    def configure(self, wbits=8, perchannel=True, sym=False, mse=False, trits=False):
        self.wbits = wbits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.trits = trits
        if trits:
            self.maxq = 3**wbits - 1
        else:
            self.maxq = 2**wbits - 1
        self.ready_flag = False

    def find_params(self, x, weight=True):
        return new_find_params(self, x, weight=weight)

    def ready(self):
        return self.ready_flag

# =============== 用到的新 quantize 函数 =================
def quantize(x, scale, zero, maxq):
    return new_quantize(x, scale, zero, maxq)

# =============== GPTQ =================
class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float16)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.float() * np.sqrt(2 / self.nsamples)
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.1, groupsize=-1, actorder=False, static_groups=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        W = W.float()

        # 调用find_params
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        lam = 10.0
        H[diag, diag] += lam * damp

        H = H.to(torch.float32)
        H = 0.5 * (H + H.transpose(-1, -2))
        try:
            H = torch.linalg.cholesky(H)
        except RuntimeError:
            print("[GPTQ] Cholesky fails, skipping layer.")
            return

        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, i1:i2].clone()
            Hinv1 = Hinv[i1:i2, i1:i2]
            Err1 = torch.zeros_like(W1)
            Q1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            count = i2 - i1
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = quantize(w.unsqueeze(1), self.quantizer.scale,
                             self.quantizer.zero, self.quantizer.maxq).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / (d**2)

                err1 = (w - q)/d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1/2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        err_total = torch.sum(Losses).item()
        print(f"[GPTQ] fasterquant finish, error = {err_total:.4f}")

        if actorder and invperm is not None:
            Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv2d):
            Q = Q.reshape(self.layer.weight.shape)
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        torch.cuda.synchronize()

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


    DEBUG = False

    # def fasterquant(
    #         self,
    #         blocksize=128,
    #         percdamp=0.1,
    #         groupsize=-1,
    #         actorder=False,
    #         static_groups=False
    # ):
    #     """
    #     结合 GPTQ 原始思路的量化示例：
    #     1) 先根据 Hessian 矩阵（self.H）判断哪些列权重重要
    #     2) 对列进行排序（actorder=True 时）
    #     3) 分块（blocksize）逐列量化 + 误差补偿
    #     4) 可选：分组量化（groupsize != -1）
    #     5) 计算量化误差并输出
    #     """
    #
    #     # ============ 0) 复制/准备权重 =============
    #     import torch
    #
    #     # 如果要支持 huggingface Transformers 的 Conv1D，需要:
    #     # import transformers
    #
    #     W = self.layer.weight.data.clone()
    #     if isinstance(self.layer, nn.Conv2d):
    #         # 对于卷积，通常先 flatten(1)
    #         W = W.flatten(1)
    #     # 若还需适配 HF Transformers.Conv1D，可做:
    #     # if isinstance(self.layer, transformers.Conv1D):
    #     #     W = W.t()
    #     W = W.float()
    #
    #     tick = time.time()
    #
    #     # 如果量化器还没 ready，则先调用 find_params(W)
    #     if not self.quantizer.ready():
    #         self.quantizer.find_params(W, weight=True)
    #         # 这步会计算出 self.quantizer.scale, self.quantizer.zero,
    #         # 即 s 和 z，这就是量化关键参数
    #
    #     # ============ 1) 取 Hessian 并做一些预处理 =============
    #     H = self.H
    #     del self.H  # 释放 self.H
    #
    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0
    #
    #     # 若使用静态分组量化，则要对每组分别 find_params
    #     # 可以参考下述逻辑，也可以改用你自己的一次性 strategy
    #     if static_groups:
    #         import copy
    #         groups = []
    #         # 逐组 clone quantizer 并计算
    #         for i in range(0, self.columns, groupsize):
    #             q_ = copy.deepcopy(self.quantizer)
    #             q_.find_params(W[:, i: i + groupsize], weight=True)
    #             groups.append(q_)
    #
    #     # ============ 2) (可选) 激活排序 actorder ============
    #     # 根据 H 的对角线大小做排序：重要列在前
    #     if actorder:
    #         perm = torch.argsort(torch.diag(H), descending=True)
    #         W = W[:, perm]
    #         H = H[perm][:, perm]
    #         invperm = torch.argsort(perm)
    #     else:
    #         perm = None
    #         invperm = None
    #
    #     # ============ 3) 构建存放量化后结果 Q、统计误差 Losses ============
    #     Q = torch.zeros_like(W)
    #     Losses = torch.zeros_like(W)
    #
    #     # percdamp=0.01 => 强力 damp
    #     damp = percdamp * torch.mean(torch.diag(H))
    #     diag = torch.arange(self.columns, device=self.dev)
    #     H[diag, diag] += damp
    #
    #     # 先对 H 做下逆或 Cholesky 之类的处理
    #     # GPTQ 原版 often 会做 Cholesky + Inverse + 重新 Cholesky
    #     H = H.to(torch.float32)
    #     H = 0.5 * (H + H.t())
    #
    #     H = torch.linalg.cholesky(H)
    #     H = torch.cholesky_inverse(H)
    #     H = torch.linalg.cholesky(H, upper=True)
    #     Hinv = H
    #
    #     # ============ 4) 分块 (blocksize) 逐列量化 + 误差补偿 ============
    #     for i1 in range(0, self.columns, blocksize):
    #         i2 = min(i1 + blocksize, self.columns)
    #         count = i2 - i1
    #
    #         W1 = W[:, i1:i2].clone()
    #         Q1 = torch.zeros_like(W1)
    #         Err1 = torch.zeros_like(W1)
    #         Losses1 = torch.zeros_like(W1)
    #
    #         Hinv1 = Hinv[i1:i2, i1:i2]
    #
    #         for i in range(count):
    #             w = W1[:, i]
    #             d = Hinv1[i, i]
    #
    #             # 如果有分组量化:
    #             if groupsize != -1:
    #                 # 计算当前列在全局 W 中的实际 idx
    #                 col_idx = i1 + i
    #
    #                 if not static_groups:
    #                     # 动态每到 groupsize 列就 find_params
    #                     if col_idx % groupsize == 0:
    #                         self.quantizer.new(
    #                             W[:, col_idx: col_idx + groupsize], weight=True
    #                         )
    #                 else:
    #                     # 如果 actorder 生效了，需要 perm 做映射
    #                     real_idx = perm[col_idx] if actorder else col_idx
    #                     # 对应组序号
    #                     group_id = real_idx // groupsize
    #                     self.quantizer = groups[group_id]
    #
    #             # ============ 量化 + 误差统计 ============
    #             q = quantize(
    #                 w.unsqueeze(1),
    #                 self.quantizer.scale,
    #                 self.quantizer.zero,
    #                 self.quantizer.maxq
    #             ).flatten()
    #
    #             Q1[:, i] = q
    #             # Loss 公式 => (w - q)^2 / d^2
    #             Losses1[:, i] = (w - q) ** 2 / (d ** 2)
    #
    #             # 误差补偿
    #             err1 = (w - q) / d
    #             # 让后续列减掉这个误差
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1
    #
    #         # 将本 block 的结果写回
    #         Q[:, i1:i2] = Q1
    #         Losses[:, i1:i2] = Losses1 / 2  # /2 仅表示 GPTQ 中常见 factor
    #
    #         # 继续更新剩余列
    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    #
    #     # 统计量化误差
    #     total_loss = torch.sum(Losses).item()
    #     print("time %.2f" % (time.time() - tick))
    #     print("error %.4f" % total_loss)
    #
    #     # ============ 5) 若 actorder => 还原列顺序 ============
    #     if actorder:
    #         Q = Q[:, invperm]
    #
    #     # 如果有 HF Conv1D => Q = Q.t()
    #     # if isinstance(self.layer, transformers.Conv1D):
    #     #     Q = Q.t()
    #
    #     # 写回量化后的权重
    #     # 恢复卷积/Conv1D时 flatten/t 转置前形状
    #     # 这里假设你只 flatten(1) 过, 还原即可
    #     self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    #
    #     if DEBUG:
    #         # 可在这里对比 self.layer(self.inp1) - self.out1 之类
    #         pass


# =============== 辅助函数: 递归找层 ===============
def find_layers(module, layers=(nn.Linear,), prefix=''):
    res = {}
    for name, mod in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(mod, layers):
            res[full_name] = mod
        else:
            res.update(find_layers(mod, layers, full_name))
    return res

# =============== GPTQ 主流程 ===============
def vita_sequential(model, calibration_loader, device='cuda'):
    model.to(device)
    model.eval()

    # 收集 nn.Linear
    layers_dict = find_layers(model, layers=(nn.Linear,))
    gptq_objs = {}
    for n, layer in layers_dict.items():
        g = GPTQ(layer)
        g.quantizer = Quantizer()
        g.quantizer.configure(wbits=8, perchannel=True, sym=False, mse=False)
        gptq_objs[n] = g

    def make_hook(nnname):
        def tmp_hook(mod, inp, out):
            gptq_objs[nnname].add_batch(inp[0], out)
        return tmp_hook

    handles = []
    for n, l in layers_dict.items():
        h = l.register_forward_hook(make_hook(n))
        handles.append(h)

    with torch.no_grad():
        for batch in calibration_loader:
            # batch["input_ids"] shape = (1,10) after squeeze
            batch["input_ids"] = batch["input_ids"].squeeze(1)
            print("calibration input_ids shape =", batch["input_ids"].shape)
            model(**{"input_ids": batch["input_ids"]})

    for h in handles:
        h.remove()

    for n in layers_dict:
        print(f"[GPTQ] start quant layer: {n}")
        gptq_objs[n].fasterquant()
        gptq_objs[n].free()

    return model

# =============== 准备一个假的校准 DataLoader ===============
def get_calibration_dataloader(file_path, device='cuda'):
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer

    class TextCalibDataset(Dataset):
        def __init__(self, file_path):
            with open(file_path, "r") as f:
                self.samples = f.readlines()  # 读取真实的文本样本
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            input_ids = self.tokenizer(self.samples[idx], padding="max_length", max_length=128, return_tensors="pt")["input_ids"]
            return {"input_ids": input_ids.squeeze(0).to(device)}

    dataset = TextCalibDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


# =============== 处理视频(保留你原先的 _get_rawvideo_dec 逻辑) ===============
def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    if not os.path.exists(video_path):
        print("Video not found:",video_path)
        return None,None

    vreader = VideoReader(video_path, ctx=cpu(0))
    fps = vreader.get_avg_fps()
    # (简化: 省略了 s,e 逻辑, 你可恢复)
    f_start=0
    f_end=len(vreader)-1
    num_frames=f_end-f_start+1
    if num_frames<=0:
        print("video length=0 ???")
        return None,None
    sample_fps=video_framerate
    t_stride=int(round(fps/sample_fps))
    all_pos=list(range(f_start,f_end+1,t_stride))
    if len(all_pos)>max_frames:
        import numpy as np
        sample_pos=[all_pos[_] for _ in np.linspace(0, len(all_pos)-1,num=max_frames,dtype=int)]
    elif len(all_pos)<min_frames:
        import numpy as np
        sample_pos=[all_pos[_] for _ in np.linspace(0,len(all_pos)-1,num=min_frames,dtype=int)]
    else:
        sample_pos=all_pos
    patch_images=[Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

    # padding, etc.
    out_imgs=[]
    for im in patch_images:
        out = image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
        out_imgs.append(out)
    patch_images=torch.stack(out_imgs)
    return patch_images, patch_images.shape[0]


# =============== 主函数(包含循环推理 100 次) ===============
def main():
    import argparse
    import os
    import time
    import torch
    from PIL import Image

    # ========== 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description="Demo with new quantize/find_params + GPTQ + multi-modal + repeated benchmark.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model dir")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")

    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--calib_data", type=str, required=True, help="Path to calibration data file")  # 校准数据文件
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--question", type=str, default="Hello, how are you?")

    parser.add_argument("--frameCat", action="store_true",
                        help="If True, uses frameCat-based image processing.")
    parser.add_argument("--disable_quant", type=int, default=0,
                        help="0=enable GPTQ, 1=skip GPTQ")

    # 循环推理次数
    parser.add_argument("--bench_rounds", type=int, default=100,
                        help="Number of repeated inferences to measure average latency.")
    args = parser.parse_args()

    # ========== 关闭 PyTorch Lazy Init（可选）==========
    from vita.util.utils import disable_torch_init
    disable_torch_init()

    # ========== 加载所需工具函数和模型构建函数 ==========
    from vita.util.mm_utils import get_model_name_from_path
    from vita.model.builder import load_pretrained_model

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"[INFO] Loading original model from: {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.model_type
    )
    model.eval().cuda()

    # ========== GPTQ 部分：量化或加载量化权重 ==========
    quant_ckpt_path = "vita_quantized.pth"
    if args.disable_quant == 0:
        print("[GPTQ] Doing calibration + quant ...")
        # 使用真实校准数据
        cali_loader = get_calibration_dataloader(args.calib_data, device="cuda")  # 需你自行实现/修改
        model = vita_sequential(model, cali_loader, "cuda")                       # 需你自行实现/已定义
        torch.cuda.empty_cache()

        # 保存量化后权重
        torch.save(model.state_dict(), quant_ckpt_path)
        print(f"[GPTQ] Quant done, saved to {quant_ckpt_path}")
    else:
        if os.path.exists(quant_ckpt_path):
            print(f"[GPTQ] Loading existing quant from {quant_ckpt_path}")
            sd = torch.load(quant_ckpt_path, map_location="cuda")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print("Missing:", missing, " Unexpected:", unexpected)
        else:
            print("[GPTQ] No quant done/loaded => using original FP weights")

    # ========== 测量模型参数量和总大小 ==========
    # 注意：如果模型是量化模型，则此时的权重已被量化，下面能体现实际 size
    param_count = sum(p.numel() for p in model.parameters())
    # 计算所有权重的字节数 (每个参数占 p.element_size() 字节)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_mb = total_bytes / (1024 ** 2)

    print(f"[INFO] Model total parameters: {param_count / 1e6:.2f} M")
    print(f"[INFO] Model param size: {param_size_mb:.2f} MB\n")

    # 打印模型中每一层参数的数据类型
    print("\n===== Model Parameters Data Types =====")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    # ========== VITA 视觉/音频模块 ==========
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        # 若需指定加载路径，则传 path=...
        vision_tower.load_model(path="/home/zqin3/vit/VITA/model_weights/InterViT-300M-448px")
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    # ========== 多模态输入处理 ==========
    if args.audio_path:
        from vita.constants import DEFAULT_AUDIO_TOKEN
        audio, audio_for_llm_lens = audio_processor.process(args.audio_path)
        audio_length = audio.shape[0]
        audio = audio.unsqueeze(0)
        audio_length = torch.tensor(audio_length).unsqueeze(0)
        audio_for_llm_lens = torch.tensor(audio_for_llm_lens).unsqueeze(0)
        audios = {
            "audios": audio.half().cuda(),
            "lengths": audio_length.half().cuda(),
            "lengths_for_llm": audio_for_llm_lens.cuda(),
        }
    else:
        audios = {
            "audios": torch.zeros((1, 400, 80), device="cuda", dtype=torch.half),
            "lengths": torch.zeros((1), device="cuda", dtype=torch.half),
            "lengths_for_llm": torch.zeros((1), device="cuda", dtype=torch.half)
        }

    from vita.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
    from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess  # 如果 frameCat=False

    modality = "lang"
    qs = args.question
    image_tensor = torch.zeros((1, 3, 448, 448), device="cuda", dtype=model.dtype)

    # 视频
    if args.video_path:
        frames, slice_len = _get_rawvideo_dec(args.video_path, image_processor)  # 需你自行实现
        if frames is not None:
            image_tensor = frames.half().cuda()
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
            if args.audio_path:
                qs += DEFAULT_AUDIO_TOKEN
            modality = "video"
    elif args.image_path:
        im = Image.open(args.image_path).convert("RGB")
        if args.frameCat:
            im, p_num = dynamic_preprocess(im, min_num=2, max_num=12, image_size=448, use_thumbnail=True)
        else:
            im, p_num = dynamic_preprocess(im, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
        image_tensor = model.process_images(im, model.config).to(dtype=model.dtype, device="cuda")

        if args.audio_path:
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
        else:
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
        modality = "image"
    else:
        # 纯文本
        if args.audio_path:
            qs = qs + DEFAULT_AUDIO_TOKEN
        modality = "lang"

    # ========== 构造 prompt & input_ids ==========
    from vita.conversation import SeparatorStyle, conv_templates
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(modality)

    from vita.util.mm_utils import (
        KeywordsStoppingCriteria,
        tokenizer_image_audio_token,
        tokenizer_image_token,
        IMAGE_TOKEN_INDEX
    )
    if args.audio_path:
        input_ids = tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    else:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # ========== 6) 重复推理 bench_rounds 次，统计平均耗时 & 峰值显存 ==========
    n_rounds = args.bench_rounds
    times = []
    peak_mem = 0.0
    final_output_ids = None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    for i in range(n_rounds):
        # 每轮推理前重置峰值统计
        torch.cuda.reset_peak_memory_stats("cuda")
        t0 = time.time()

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=0.01,
                top_p=None,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                shared_v_pid_stride=None
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)

        final_output_ids = out.sequences
        mem_now = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)
        if mem_now > peak_mem:
            peak_mem = mem_now

    avg_time = sum(times) / len(times)

    # ========== 7) 解析输出 + 打印结果 ==========
    input_token_len = input_ids.shape[1]
    output_ids = final_output_ids

    # 如果是 mixtral-8x7b 则尝试去掉输入 prompt
    if args.model_type == "mixtral-8x7b":
        # 注意：若 output_ids 太短，可能会报 shape error，需做长度判断
        if output_ids.shape[1] >= input_token_len:
            n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff > 0:
                print(f"[Warning] {n_diff} output_ids differ from input_ids => slicing")
                output_ids = output_ids[:, input_token_len:]
        else:
            print("[Warning] Output length < input length; skip slicing.")

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    print("\n===== Benchmark Results =====")
    print(f"Rounds: {n_rounds}")
    print(f"Average inference time: {avg_time:.4f} s")
    print(f"Peak GPU mem usage: {peak_mem:.1f} MB")

    # print(f"Model param count: {param_count} (~{param_count/1e6:.2f}M)")
    # print(f"Model param size: {param_size_mb:.2f} MB (on GPU, after quant if any)")

    print("\n===== Last round LLM output =====")
    print(outputs)




if __name__ == "__main__":
    main()


