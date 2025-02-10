# 这是video_audio_memory.py
# !/usr/bin/env python3
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

def new_quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero

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

    self.ready_flag = True

    # perchannel 完成后还原 shape
    if weight:
        shape2 = [-1] + [1] * (len(shape)-1)
        self.scale = self.scale.reshape(shape2)
        self.zero  = self.zero.reshape(shape2)
    else:
        pass

# =============== Quantizer 类 ===============
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

# =============== quantize =================
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

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.1,
            groupsize=-1,
            actorder=False,
            static_groups=False
    ):
        import torch

        # 如果要支持 huggingface Transformers 的 Conv1D，需要:
        # import transformers

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            # 对于卷积，通常先 flatten(1)
            W = W.flatten(1)
        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
            # 计算出 self.quantizer.scale, self.quantizer.zero

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            # 逐组 clone quantizer 并计算
            for i in range(0, self.columns, groupsize):
                q_ = copy.deepcopy(self.quantizer)
                q_.find_params(W[:, i: i + groupsize], weight=True)
                groups.append(q_)

        # ============激活排序 actorder ============
        # 根据 H 的对角线大小做排序：重要列在前
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        # ============ 存放量化后结果 Q、error, Losses ============
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

        # ============ 逐列量化 ============
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # 如果有分组量化:
                if groupsize != -1:
                    # 计算当前列在全局 W 中的实际 idx
                    col_idx = i1 + i

                    if not static_groups:
                        # 动态每到 groupsize 列就 find_params
                        if col_idx % groupsize == 0:
                            self.quantizer.new_find_params(
                                W[:, col_idx: col_idx + groupsize], weight=True
                            )
                    else:
                        real_idx = perm[col_idx] if actorder else col_idx
                        group_id = real_idx // groupsize
                        self.quantizer = groups[group_id]

                # ============ 量化误差统计 ============
                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq
                ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / (d ** 2)

                # 误差补偿
                err1 = (w - q) / d
                # 让后续列减掉这个误差
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            # 将本 block 的结果写回
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # 继续更新剩余列
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # 统计量化误差
        total_loss = torch.sum(Losses).item()
        print("time %.2f" % (time.time() - tick))
        print("error %.4f" % total_loss)

        if actorder:
            Q = Q[:, invperm]

        # 如果有 HF Conv1D => Q = Q.t()
        # if isinstance(self.layer, transformers.Conv1D):
        #     Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if actorder and invperm is not None:
            Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv2d):
            Q = Q.reshape(self.layer.weight.shape)
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        torch.cuda.synchronize()

        DEBUG = False


def find_layers(module, layers=(nn.Linear,), prefix=''):
    res = {}
    for name, mod in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(mod, layers):
            res[full_name] = mod
        else:
            res.update(find_layers(mod, layers, full_name))
    return res

# =============== GPTQ ===============
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

# =============== DataLoader ===============
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


# =============== 处理视频 ===============
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


# =============== 主函数===============
def main():
    import argparse
    import os
    import time
    import torch
    from PIL import Image

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

    from vita.util.utils import disable_torch_init
    disable_torch_init()

    from vita.util.mm_utils import get_model_name_from_path
    from vita.model.builder import load_pretrained_model

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"[INFO] Loading original model from: {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.model_type
    )
    model.eval().cuda()

    # ========== GPTQ ==========
    quant_ckpt_path = "vita_quantized.pth"
    if args.disable_quant == 0:
        print("[GPTQ] Doing calibration + quant ...")
        # 使用校准数据
        cali_loader = get_calibration_dataloader(args.calib_data, device="cuda")
        model = vita_sequential(model, cali_loader, "cuda")
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
    param_count = sum(p.numel() for p in model.parameters())
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
        # 指定加载路径
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
        frames, slice_len = _get_rawvideo_dec(args.video_path, image_processor)
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

    # ========== 重复推理统计平均耗时 ==========
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

    input_token_len = input_ids.shape[1]
    output_ids = final_output_ids

    # 如果是 mixtral-8x7b 则尝试去掉输入 prompt
    if args.model_type == "mixtral-8x7b":
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


