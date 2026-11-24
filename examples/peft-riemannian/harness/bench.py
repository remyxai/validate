"""LoRA teacher-student benchmark for an optimizer change in PEFT.

A frozen two-layer MLP gets a hidden low-rank update (the "teacher"). A LoRA
adapter on the same frozen MLP is trained to match the teacher's outputs.
The only thing that differs between arms is the optimizer:

    OPTIMIZER=adamw        torch.optim.AdamW on the LoRA parameters
    OPTIMIZER=riemannian   peft.optimizers.create_riemannian_optimizer(AdamW)

Everything is seeded from $VALIDATE_SEED, so both arms see the same base
weights, teacher, LoRA init and batches. CPU only, no downloads.
"""
import json
import os
import time

import torch
from torch import nn

from peft import LoraConfig, get_peft_model

SEED = int(os.environ.get("VALIDATE_SEED", "0"))
OPTIMIZER = os.environ.get("OPTIMIZER", "adamw")
STEPS = int(os.environ.get("BENCH_STEPS", "400"))
LR = float(os.environ.get("BENCH_LR", "2e-3"))
RANK = int(os.environ.get("BENCH_RANK", "8"))
SMOKE = os.environ.get("VALIDATE_SMOKE") == "1"
if SMOKE:
    STEPS = 5

D_IN, D_HID, D_OUT, TEACHER_RANK, BATCH = 64, 128, 32, 4, 64


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(D_IN, D_HID)
        self.act = nn.Tanh()
        self.lin2 = nn.Linear(D_HID, D_OUT)

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


def main():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    base = MLP()
    teacher = MLP()
    teacher.load_state_dict(base.state_dict())
    with torch.no_grad():
        for layer in (teacher.lin1, teacher.lin2):
            out_f, in_f = layer.weight.shape
            u = torch.randn(out_f, TEACHER_RANK, generator=gen)
            v = torch.randn(TEACHER_RANK, in_f, generator=gen)
            layer.weight += 0.5 * (u @ v) / (in_f ** 0.5 * TEACHER_RANK ** 0.5)
    teacher.requires_grad_(False)

    eval_x = torch.randn(2048, D_IN, generator=gen)
    eval_y = teacher(eval_x)

    torch.manual_seed(SEED)  # LoRA init is the same in both arms
    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["lin1", "lin2"]))
    trainable = [p for p in model.parameters() if p.requires_grad]

    if OPTIMIZER == "riemannian":
        from peft.optimizers import create_riemannian_optimizer
        opt = create_riemannian_optimizer(model, torch.optim.AdamW, lr=LR, weight_decay=0.0)
    elif OPTIMIZER == "adamw":
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    else:
        raise SystemExit(f"unknown OPTIMIZER {OPTIMIZER!r}")

    loss_fn = nn.MSELoss()
    with torch.no_grad():
        initial = loss_fn(model(eval_x), eval_y).item()

    start = time.perf_counter()
    for _ in range(STEPS):
        x = torch.randn(BATCH, D_IN, generator=gen)
        loss = loss_fn(model(x), teacher(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - start

    with torch.no_grad():
        final = loss_fn(model(eval_x), eval_y).item()

    metrics = {
        "final_eval_loss": final,
        "initial_eval_loss": initial,
        "loss_reduction": 1.0 - final / initial,
        "steps_per_s": STEPS / train_s,
        "num_trainable_params": sum(p.numel() for p in trainable),
        "train_steps": STEPS,
        # Witness that the candidate arm really got a preconditioned optimizer.
        "optimizer_preconditioned": int("Riemannian" in type(opt).__name__),
    }
    out_dir = os.environ.get("VALIDATE_OUTPUT_DIR")
    if out_dir:
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump({"metrics": metrics}, fh)
    print("METRICS_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
