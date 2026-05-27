from typing import Optional

import torch

from .rl_diff import RL_Lightning


class RL_GRPO_Surrogate_Lightning(RL_Lightning):
    """GRPO-style flow-matching finetuning with switchable surrogate objectives.

    surrogate_mode options:
      - "single_time_surrogate": one sampled time per batch element.
      - "multi_time_surrogate": multiple sampled times per batch element, averaged.
      - "trajectory_surrogate": fixed trajectory-like time grid, averaged.

    For each generated batch, old surrogate losses are cached once and the model is
    updated for K inner iterations using ratio-style weighting:
      ratio = exp(l_old - l_new)
      objective ~ E[ratio * advantage]
    """

    def __init__(
        self,
        gen: torch.nn.Module,
        vocab,
        lr: float,
        coord_scale: float = 1.0,
        use_ema: bool = True,
        compile_model: bool = True,
        warm_up_steps: Optional[int] = None,
        max_steps: int = 128,
        default_coord_noise_std: float = 0.2,
        default_cat_noise_level: float = 1.0,
        self_cond: bool = False,
        loss_weight: dict = {"types": 0.2, "bonds": 1.0, "charges": 1.0},
        formulation: str = "endpoint",
        eval_3D_props: bool = True,
        reward_name: str = "qed",
        reward_beta: float = 2.0,
        reward_weight_min: float = 0.1,
        reward_weight_max: float = 10.0,
        reward_norm_eps: float = 1e-6,
        anchor_weight: float = 0.1,
        anchor_loss_weight: float = 1.0,
        use_reference_anchor: bool = True,
        dynamic_anchor: bool = True,
        target_anchor_loss: float = 0.05,
        anchor_update_rate_up: float = 1.5,
        anchor_update_rate_down: float = 0.98,
        anchor_weight_min: float = 1e-4,
        anchor_weight_max: float = 10.0,
        anchor_warmup_steps: int = 100,
        anchor_ema_momentum: float = 0.95,
        use_ema_reference: bool = True,
        ema_reference_decay: float = 0.995,
        ema_reference_update_every: int = 1,
        surrogate_mode: str = "ddpo_sf",
        adv_clip: float = 5.0,
        multi_time_samples: int = 4,
        k_updates: int = 2,
        clip_eps: float = 0.2,
        grad_clip_val: Optional[float] = 1.0,
    ):
        super().__init__(
            gen=gen,
            vocab=vocab,
            lr=lr,
            coord_scale=coord_scale,
            use_ema=use_ema,
            compile_model=compile_model,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            default_coord_noise_std=default_coord_noise_std,
            default_cat_noise_level=default_cat_noise_level,
            self_cond=self_cond,
            loss_weight=loss_weight,
            formulation=formulation,
            eval_3D_props=eval_3D_props,
            reward_name=reward_name,
            reward_beta=reward_beta,
            reward_weight_min=reward_weight_min,
            reward_weight_max=reward_weight_max,
            reward_norm_eps=reward_norm_eps,
            anchor_weight=anchor_weight,
            anchor_loss_weight=anchor_loss_weight,
            use_reference_anchor=use_reference_anchor,
        )

        valid_modes = {"ddpo_sf", "ddpo_is"}
        if surrogate_mode not in valid_modes:
            raise ValueError(f"Unsupported surrogate_mode='{surrogate_mode}'. Valid: {sorted(valid_modes)}")

        self.surrogate_mode = surrogate_mode
        self.adv_clip = adv_clip
        self.multi_time_samples = max(1, int(multi_time_samples))
        self.k_updates = max(1, int(k_updates))
        self.clip_eps = clip_eps
        self.grad_clip_val = grad_clip_val
        self.dynamic_anchor = dynamic_anchor
        self.target_anchor_loss = target_anchor_loss
        self.anchor_update_rate_up = anchor_update_rate_up
        self.anchor_update_rate_down = anchor_update_rate_down
        self.anchor_weight_min = anchor_weight_min
        self.anchor_weight_max = anchor_weight_max
        self.anchor_warmup_steps = anchor_warmup_steps
        self.anchor_ema_momentum = anchor_ema_momentum
        self.use_ema_reference = use_ema_reference
        self.ema_reference_decay = ema_reference_decay
        self.ema_reference_update_every = max(1, int(ema_reference_update_every))
        self.anchor_loss_ema = None
        self.rl_inner_step = 0

        self.automatic_optimization = False

    def _update_anchor_weight(self, anchor_loss: torch.Tensor):
        if not self.dynamic_anchor:
            return
        if self.global_step < self.anchor_warmup_steps:
            return

        anchor_val = float(anchor_loss.detach().item())
        if self.anchor_loss_ema is None:
            self.anchor_loss_ema = anchor_val
        else:
            m = self.anchor_ema_momentum
            self.anchor_loss_ema = m * self.anchor_loss_ema + (1 - m) * anchor_val

        if self.anchor_loss_ema > self.target_anchor_loss * 1.5:
            self.anchor_weight *= self.anchor_update_rate_up
        elif self.anchor_loss_ema < self.target_anchor_loss * 0.5:
            self.anchor_weight *= self.anchor_update_rate_down

        self.anchor_weight = max(self.anchor_weight_min, min(self.anchor_weight, self.anchor_weight_max))

    @torch.no_grad()
    def _update_ema_reference(self):
        if not self.use_reference_anchor:
            return
        if self.ref_gen is None:
            return
        if not self.use_ema_reference:
            return

        decay = self.ema_reference_decay
        for ref_param, param in zip(self.ref_gen.parameters(), self.gen.parameters()):
            ref_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

        for ref_buf, buf in zip(self.ref_gen.buffers(), self.gen.buffers()):
            ref_buf.data.copy_(buf.data)

    def _standardized_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        centered = rewards - rewards.mean()
        adv = centered / (centered.std(unbiased=False) + self.reward_norm_eps)
        return torch.clamp(adv, min=-self.adv_clip, max=self.adv_clip)

    def _loss_components_at_t(self, train_batch, t, flag_3Ds):
        interp_data = self.interpolate(train_batch, t, flag_3Ds=flag_3Ds)

        cond_batch = None
        if self.self_cond:
            cond_batch = {
                "coords": torch.zeros_like(interp_data["coords"]),
                "atomics": torch.zeros_like(interp_data["atomics"]),
                "bonds": torch.zeros_like(interp_data["bonds"]),
            }


            # if torch.rand(1).item() > 0.5:
            #     with torch.no_grad():
            #         cond_coords, cond_types, cond_bonds, _ = self(
            #             interp_data,
            #             t,
            #             training=True,
            #             cond_batch=cond_batch,
            #             flag_3Ds=flag_3Ds,
            #         )

            #         cond_batch = {
            #             "coords": cond_coords * flag_3Ds.view(-1, 1, 1),
            #             "atomics": torch.softmax(cond_types, dim=-1),
            #             "bonds": torch.softmax(cond_bonds, dim=-1),
            #         }
                    
        coords, types, bonds, charges = self(
            interp_data,
            t,
            training=True,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )

        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        if self.formulation == "endpoint":
            coords_target = train_batch["real_coords"]
        else:
            coords_target = train_batch["real_coords"] - train_batch["noise_coords"]

        target = {
            "coords": coords_target,
            "atomics": train_batch["real_atomics"],
            "bonds": train_batch["real_bonds"],
            "charges": train_batch["real_charges"],
            "masks": train_batch["masks"],
        }

        losses = self._loss_per_sample(target, predicted, flag_3Ds=flag_3Ds)

        anchor_losses = self._anchor_loss_per_sample(
            interp_data,
            t,
            predicted,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )
        anchor_loss = (
            anchor_losses["coord"]
            + anchor_losses["types"]
            + anchor_losses["bonds"]
            + anchor_losses["charges"]
        )

        return losses, anchor_loss


    def _ratio_objective(self, l_old, l_new, advantages):
        ratio = torch.exp(l_old - l_new)
        ratio_clipped = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        loss_ratio = -torch.min(surr1, surr2).mean()
        return loss_ratio, ratio.mean().detach()

    def FM_training_step(self, batch):
        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )

        rewards, generated_mols = self._compute_rewards_from_generated(generated)
        quality_metrics = self._compute_generation_quality_from_mols(
            generated_mols,
            dtype=generated["coords"].dtype,
            device=generated["coords"].device,
        )

        weights = self._reward_to_weights(rewards)
        advantages = weights / weights.mean().clamp_min(self.reward_norm_eps)
        train_batch = self._build_generated_target_batch(batch, generated)
        flag_3Ds = train_batch["flag_3Ds"]
        batchsize = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device

        t = self.time_dist.sample((batchsize,)).to(device)

        if self.surrogate_mode == "ddpo_sf":
            old_losses, old_anchor_loss = self._loss_components_at_t(train_batch, t, flag_3Ds)
            opt = self.optimizers()
            opt.zero_grad()
            old_loss = sum(old_losses.values())
            fm_loss = -(torch.exp(-old_loss) * advantages).mean()
            total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * old_anchor_loss.mean()
            self.manual_backward(total_loss)
            if self.grad_clip_val is not None and self.grad_clip_val > 0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.grad_clip_val,
                    gradient_clip_algorithm="norm",
                )
            opt.step()
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()
        elif self.surrogate_mode == "ddpo_is":
            with torch.no_grad():
                old_losses, old_anchor_loss = self._loss_components_at_t(train_batch, t, flag_3Ds)
                old_losses = {k: v.detach() for k, v in old_losses.items()}
                old_loss = sum(old_losses.values())

            for _ in range(self.k_updates):
                losses, anchor_loss = self._loss_components_at_t(train_batch, t, flag_3Ds)
                new_loss = sum(losses.values())

                opt = self.optimizers()
                
                opt.zero_grad()
                fm_loss = -(torch.exp(old_loss-new_loss) * advantages).mean()
                
                total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * anchor_loss.mean()
                self.manual_backward(total_loss)
                if self.grad_clip_val is not None and self.grad_clip_val > 0:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.grad_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                opt.step()
                sch = self.lr_schedulers()
                if sch is not None:
                    sch.step()


        mode_index = {
            "ddpo_sf": 0,
            "ddpo_is": 1,
        }[self.surrogate_mode]

        self.log("train-rl-surrogate-mode", float(mode_index), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-adv-mean", advantages.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-adv-std", advantages.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
        # self.log("train-rl-ratio", r_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-anchor-weight", float(self.anchor_weight), on_step=True, logger=True, sync_dist=True)
        if self.anchor_loss_ema is not None:
            self.log("train-rl-anchor-loss-ema", float(self.anchor_loss_ema), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-ema-ref-decay", float(self.ema_reference_decay), on_step=True, logger=True, sync_dist=True)

        self.log("train-rl-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        # self.log("train-rl-anchor-loss", final_anchor_loss, on_step=True, logger=True, sync_dist=True)
        # self.log("train-rl-fm-loss", final_fm_loss, on_step=True, logger=True, sync_dist=True)
        # self.log("train-rl-total-loss", final_total_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)

        return total_loss
