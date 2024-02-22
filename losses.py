from typing import List, Tuple
import torch


def feature_loss(
    fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
) -> torch.Tensor:
    loss = fmap_r[0][0].new_tensor(0)  # device trick from gpt
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss = disc_real_outputs[0].new_tensor(0)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]):
    loss = disc_outputs[0].new_tensor(0)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
