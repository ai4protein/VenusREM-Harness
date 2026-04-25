from typing import Optional

import torch

from src.orbit.fusions.base import BaseFusion
from src.orbit.types import RetrievalOutput


class TwoStageLearnableGateFusion(BaseFusion):
    """
    Stage-1: fuse PLM and legacy retrieval logits.
    Stage-2: fuse Stage-1 logits with PSALOR layer2 logits via confidence gate.
    """

    def __init__(
        self,
        stage1_fusion: BaseFusion,
        layer2_weight: float = 1.0,
        gate_temperature: float = 1.0,
        alpha_family: float = 1.0,
        enable_gate_diagnostics: bool = False,
    ) -> None:
        self.stage1_fusion = stage1_fusion
        self.layer2_weight = float(layer2_weight)
        self.gate_temperature = float(gate_temperature)
        self.alpha_family = float(alpha_family)
        self.enable_gate_diagnostics = bool(enable_gate_diagnostics)

    def _print_gate_stats(self, gate: torch.Tensor, retrieval_output: RetrievalOutput) -> None:
        if not self.enable_gate_diagnostics:
            return
        flat = gate.reshape(-1).float()
        if flat.numel() == 0:
            return
        mean_v = float(flat.mean().item())
        q10 = float(torch.quantile(flat, 0.10).item())
        q90 = float(torch.quantile(flat, 0.90).item())
        protein_name = retrieval_output.metadata.get("protein_name", "unknown")
        layer2_name = retrieval_output.layer2_name or "layer2"
        print(
            f"[gate-diagnostic] protein={protein_name} layer2={layer2_name} "
            f"mean={mean_v:.4f} p10={q10:.4f} p90={q90:.4f}"
        )

    def _layer2_gate(self, layer2_logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.tensor(
            max(self.gate_temperature, 1e-3), dtype=torch.float32, device=layer2_logits.device
        )
        probs = torch.softmax(layer2_logits / temperature, dim=-1)
        confidence = probs.max(dim=-1).values.unsqueeze(-1)
        base = torch.tensor(
            self.layer2_weight, dtype=torch.float32, device=layer2_logits.device
        ).clamp(0.0, 1.0)
        fam = torch.tensor(
            self.alpha_family, dtype=torch.float32, device=layer2_logits.device
        ).clamp(0.0, 1.0)
        gate = base * fam * confidence
        return gate.clamp(0.0, 1.0)

    def fuse(
        self,
        plm_logits: torch.Tensor,
        retrieval_output: RetrievalOutput,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        stage1_logits = self.stage1_fusion.fuse(
            plm_logits=plm_logits, retrieval_output=retrieval_output, mode=mode
        )
        if retrieval_output.layer2_logits is None:
            return stage1_logits

        layer2_logits = retrieval_output.layer2_logits.to(stage1_logits.device)
        gate = self._layer2_gate(layer2_logits).to(stage1_logits.device)
        if retrieval_output.layer2_position_weights is not None:
            pos_w = retrieval_output.layer2_position_weights.to(stage1_logits.device)
            gate = gate * pos_w
        if retrieval_output.layer2_mask is not None:
            gate = gate * retrieval_output.layer2_mask.to(stage1_logits.device)
        gate = gate.clamp(0.0, 1.0)
        self._print_gate_stats(gate, retrieval_output)
        return (1.0 - gate) * stage1_logits + gate * layer2_logits
