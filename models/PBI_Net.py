"""
PBI-Net: Domain-informed Periodic-Background Interaction Network.

This model is designed for hourly gravity time-series forecasting.

Method overview
---------------
1. Preserve domain-informed 12 h / 24 h harmonic priors with fixed
   Gaussian frequency bands.
2. Explicitly decompose the input spectrum into:
      (a) periodic spectrum; and
      (b) background spectrum.
3. Encode the two domain-informed spectral components independently.
4. Model their nonlinear interaction with a sequential affine update whose
   topology follows the published CFPT formulation.
5. Predict a background spectrum and a periodic residual, combine them
   adaptively in the frequency domain, and reconstruct the future signal with
   an inverse real FFT.

Important attribution
---------------------
The sequential affine update topology is not claimed as a new contribution;
the implementation below expresses the published equations directly in
PBI-Net's periodic/background terminology. The model's main formulation is the
gravity-specific domain-informed 12/24 h prior, explicit periodic/background
spectral decomposition, and interaction of these two domain-informed spectral
representations.

The public ``Model`` interface follows Time-Series-Library:

    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

Only ``x_enc`` is required. The default configuration matches the current
Gravity/Membach/Medicina/Djougou experiments (seq_len=96, pred_len=24,
periods=12,24).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


def _parse_periods(value) -> Tuple[float, ...]:
    """Accept ``"12,24"``, a scalar, or an iterable of period lengths."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        periods = tuple(float(item) for item in items)
    elif isinstance(value, (int, float)):
        periods = (float(value),)
    elif isinstance(value, Iterable):
        periods = tuple(float(item) for item in value)
    else:
        raise TypeError(
            "periods must be a comma-separated string, scalar, or iterable"
        )

    if not periods:
        raise ValueError("At least one domain-informed period must be specified")
    if len(set(periods)) != len(periods):
        raise ValueError(f"Duplicate periods are not allowed: {periods}")
    for period in periods:
        if period <= 2.0:
            raise ValueError(
                f"Each period must be greater than 2 samples, got {period}"
            )
    return periods


def _parse_center_bins(value) -> Tuple[int, ...]:
    """Accept an empty value or comma-separated positive integer FFT bins."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        bins = tuple(int(item) for item in items)
    elif isinstance(value, int):
        bins = (int(value),)
    elif isinstance(value, Iterable):
        bins = tuple(int(item) for item in value)
    else:
        raise TypeError(
            "center bins must be an empty value, comma-separated string, "
            "integer, or iterable"
        )

    if len(set(bins)) != len(bins):
        raise ValueError(f"Duplicate center bins are not allowed: {bins}")
    if any(center <= 0 for center in bins):
        raise ValueError(f"FFT center bins must be positive, got {bins}")
    return bins


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _build_affine_conditioner(
    branch_width: int,
    latent_width: int,
) -> nn.Sequential:
    """Build one 64->128->64 conditioner used by an affine update."""
    return nn.Sequential(
        nn.Linear(branch_width, latent_width),
        nn.LayerNorm(latent_width),
        nn.ReLU(inplace=False),
        nn.Linear(latent_width, branch_width),
    )


class InteractionBlock(nn.Module):
    """
    Sequential affine interaction between fixed semantic latent branches.

    In PBI-Net, the first half of the latent vector is always the background
    representation and the second half is always the domain-informed periodic
    representation. The update topology follows the published CFPT equations,
    while this implementation is organized directly around those PBI-Net
    branch semantics.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("InteractionBlock channels must be even")

        self.channels = int(channels)
        self.channels_half = self.channels // 2

        # Keep the historical registration names and order so archived PBI-Net
        # checkpoints remain loadable without a key conversion.
        self.s1 = _build_affine_conditioner(
            self.channels_half,
            self.channels,
        )
        self.t1 = _build_affine_conditioner(
            self.channels_half,
            self.channels,
        )
        self.s2 = _build_affine_conditioner(
            self.channels_half,
            self.channels,
        )
        self.t2 = _build_affine_conditioner(
            self.channels_half,
            self.channels,
        )

        # Small initial scale keeps the exponential affine modulation stable.
        self.scale_1 = nn.Parameter(torch.tensor(0.1))
        self.scale_2 = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _conditioned_affine_update(
        target: torch.Tensor,
        conditioner: torch.Tensor,
        scale_network: nn.Module,
        shift_network: nn.Module,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``target * exp(clip(gain*S(conditioner))) + T``."""
        log_scale = scale_network(conditioner)
        shift = shift_network(conditioner)
        bounded_log_scale = torch.clamp(
            gain * log_scale,
            min=-5.0,
            max=5.0,
        )
        return target * torch.exp(bounded_log_scale) + shift

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"Expected last dimension {self.channels}, got {x.shape[-1]}"
            )

        background = x[..., : self.channels_half]
        periodic = x[..., self.channels_half :]

        # Step 1: periodic-conditioned background update.
        background_updated = self._conditioned_affine_update(
            target=background,
            conditioner=periodic,
            scale_network=self.s1,
            shift_network=self.t1,
            gain=self.scale_1,
        )

        # Step 2: background-conditioned periodic update. The updated output of
        # Step 1 is intentionally used as the conditioner.
        periodic_updated = self._conditioned_affine_update(
            target=periodic,
            conditioner=background_updated,
            scale_network=self.s2,
            shift_network=self.t2,
            gain=self.scale_2,
        )

        return torch.concat(
            (background_updated, periodic_updated),
            dim=-1,
        )


class ResidualMLPFusionBlock(nn.Module):
    """Generic capacity-matched fusion control for the CFI comparison."""

    def __init__(self, channels: int, hidden: int = 261):
        super().__init__()
        if channels < 1 or hidden < 1:
            raise ValueError("channels and hidden must be positive")
        self.fc1 = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor):
        return x + self.fc2(self.activation(self.fc1(x)))


class FixedPhysicalPeriodicPrior(nn.Module):
    """
    Fixed Gaussian frequency masks around domain-informed periods.

    For hourly gravity observations, 12 h and 24 h correspond to the main
    semi-diurnal and diurnal prior bands used in the experiments.

    All configured period bands are always active at full strength, their
    Gaussian masks are summed, and overlaps are clipped to [0, 1]. The prior is
    used only for input-side periodic/background decomposition; no future
    frequency mask is imposed.
    """

    def __init__(
        self,
        seq_len: int,
        periods: Sequence[float],
        bandwidth: float = 1.0,
        center_bins: Sequence[int] = (),
        mask_mode: str = "physical",
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        configured_periods = tuple(float(period) for period in periods)
        self.center_bins = tuple(int(center) for center in center_bins)
        self.bandwidth = float(bandwidth)
        self.mask_mode = str(mask_mode).strip().lower()

        if self.seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.mask_mode not in {"physical", "neutral"}:
            raise ValueError(
                "mask_mode must be 'physical' or 'neutral', "
                f"got {self.mask_mode!r}"
            )

        max_bin = self.seq_len // 2
        if any(center > max_bin for center in self.center_bins):
            raise ValueError(
                f"FFT center bins must be in [1, {max_bin}], "
                f"got {self.center_bins}"
            )

        if self.center_bins:
            self.periods = tuple(
                self.seq_len / float(center) for center in self.center_bins
            )
        else:
            self.periods = configured_periods

        period_tensor = torch.tensor(self.periods, dtype=torch.float32)
        self.register_buffer("period_tensor", period_tensor)

        if self.center_bins:
            center_bin_tensor = torch.tensor(
                self.center_bins,
                dtype=torch.float32,
            )
        else:
            # Preserve the authoritative full-model computation exactly.
            center_bin_tensor = self.seq_len / period_tensor
        # Non-persistent so authoritative full checkpoints keep identical keys.
        self.register_buffer(
            "center_bin_tensor",
            center_bin_tensor,
            persistent=False,
        )

        input_band_bank = self._make_band_bank(
            signal_len=self.seq_len,
            period_tensor=period_tensor,
            bandwidth=self.bandwidth,
            center_bin_tensor=(
                center_bin_tensor if self.center_bins else None
            ),
        )

        self.register_buffer("input_band_bank", input_band_bank)
        if self.mask_mode == "neutral":
            # Isomorphic no-prior control: both branches receive one half of
            # every rFFT coefficient.  There is no frequency-position
            # assignment, while the dual encoders, CFI, heads and parameter
            # count remain unchanged.
            input_period_mask = torch.full(
                (self.seq_len // 2 + 1,),
                0.5,
                dtype=input_band_bank.dtype,
            )
        else:
            input_period_mask = self._combine_full_strength_bands(
                input_band_bank
            )
        self.register_buffer("input_period_mask", input_period_mask)

    @staticmethod
    def _make_band_bank(
        signal_len: int,
        period_tensor: torch.Tensor,
        bandwidth: float,
        center_bin_tensor: torch.Tensor | None = None,
    ):
        freq_bins = torch.arange(
            signal_len // 2 + 1,
            dtype=torch.float32,
        )
        centers = (
            signal_len / period_tensor
            if center_bin_tensor is None
            else center_bin_tensor
        )
        distance = (
            freq_bins.unsqueeze(0) - centers.unsqueeze(1)
        ) / bandwidth

        bands = torch.exp(-0.5 * distance.square())
        bands[:, 0] = 0.0  # DC always belongs to the background branch.
        return bands

    @staticmethod
    def _combine_full_strength_bands(band_bank: torch.Tensor):
        return band_bank.sum(dim=0).clamp(0.0, 1.0)


class PBIFrequencyModel(nn.Module):
    """Frequency-domain backbone of PBI-Net."""

    def __init__(self, configs):
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.d_model = int(configs.d_model)
        self.e_layers = int(configs.e_layers)
        self.ablation = str(getattr(configs, 'pbi_ablation', 'full')).lower()

        valid_ablation_modes = {'full', 'no_cfi', 'matched_mlp'}
        if self.ablation not in valid_ablation_modes:
            raise ValueError(
                f'Unsupported pbi_ablation={self.ablation!r}; '
                f'choose from {sorted(valid_ablation_modes)}'
            )

        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for periodic/background CFI")
        if self.e_layers < 1:
            raise ValueError("e_layers must be at least 1")

        # Reuse the SAP-CFI command-line option names when present so the model
        # uses the common forecasting entry point without extra parser options.
        periods = _parse_periods(
            getattr(configs, "sap_periods", "12,24")
        )
        bandwidth = float(getattr(configs, "sap_bandwidth", 1.0))
        center_bins = _parse_center_bins(
            getattr(configs, "pbi_center_bins", "")
        )
        mask_mode = str(
            getattr(configs, "pbi_mask_mode", "physical")
        ).strip().lower()

        self.prior = FixedPhysicalPeriodicPrior(
            seq_len=self.seq_len,
            periods=periods,
            bandwidth=bandwidth,
            center_bins=center_bins,
            mask_mode=mask_mode,
        )

        self.freq_len = self.seq_len // 2 + 1
        self.freq_dims = self.freq_len * 2
        self.branch_dim = self.d_model // 2

        # Independent encoders preserve the semantic distinction between
        # background and domain-informed periodic spectra before interaction.
        self.background_embedding = nn.Linear(
            self.freq_dims,
            self.branch_dim,
        )
        self.periodic_embedding = nn.Linear(
            self.freq_dims,
            self.branch_dim,
        )

        # The formal model uses CFI. ``matched_mlp`` replaces, rather than
        # supplements, CFI with equally many generic residual MLP blocks.
        if self.ablation == 'matched_mlp':
            mlp_hidden = int(getattr(configs, 'pbi_mlp_hidden', 261))
            self.interaction_blocks = nn.ModuleList(
                [
                    ResidualMLPFusionBlock(
                        self.d_model,
                        hidden=mlp_hidden,
                    )
                    for _ in range(self.e_layers)
                ]
            )
        else:
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionBlock(self.d_model)
                    for _ in range(self.e_layers)
                ]
            )

        self.pred_freq_len = self.pred_len // 2 + 1
        self.pred_freq_dims = self.pred_freq_len * 2

        self.background_predictor = nn.Linear(
            self.branch_dim,
            self.pred_freq_dims,
        )
        self.periodic_residual_predictor = nn.Linear(
            self.branch_dim,
            self.pred_freq_dims,
        )

        zero_init = _as_bool(
            getattr(configs, "sap_zero_init_period_head", True)
        )
        if zero_init:
            nn.init.zeros_(self.periodic_residual_predictor.weight)
            nn.init.zeros_(self.periodic_residual_predictor.bias)

        self.last_diagnostics = None

    @staticmethod
    def _complex_to_real_features(spectrum: torch.Tensor):
        return torch.cat([spectrum.real, spectrum.imag], dim=-1)

    def _real_features_to_complex(self, features: torch.Tensor):
        real_part, imag_part = torch.split(
            features,
            self.pred_freq_len,
            dim=-1,
        )
        return torch.complex(real_part, imag_part)

    def forward(self, x_enc: torch.Tensor):
        if x_enc.ndim != 3:
            raise ValueError(
                "x_enc must have shape [B, L, N], "
                f"got {tuple(x_enc.shape)}"
            )

        batch_size, length, channels = x_enc.shape
        if length != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {length}"
            )

        # --------------------------------------------------------------
        # 1) Frequency transform
        # --------------------------------------------------------------
        x_channels_first = x_enc.transpose(1, 2)  # [B, N, L]
        full_spectrum = torch.fft.rfft(x_channels_first, dim=-1)

        # --------------------------------------------------------------
        # 2) Fixed domain-informed 12/24 h periodic-background decomposition
        # --------------------------------------------------------------
        input_period_mask = self.prior.input_period_mask.to(
            dtype=x_enc.dtype,
            device=x_enc.device,
        )

        periodic_spectrum = full_spectrum * input_period_mask
        background_spectrum = full_spectrum - periodic_spectrum

        # --------------------------------------------------------------
        # 3) Independent branch encoding
        # --------------------------------------------------------------
        background_latent = self.background_embedding(
            self._complex_to_real_features(background_spectrum)
        )
        periodic_latent = self.periodic_embedding(
            self._complex_to_real_features(periodic_spectrum)
        )

        # Background is concatenated first, periodic second. InteractionBlock
        # therefore always operates on semantically fixed branch positions.
        latent = torch.cat(
            [background_latent, periodic_latent],
            dim=-1,
        )
        latent = latent.reshape(
            batch_size * channels,
            self.d_model,
        )

        # --------------------------------------------------------------
        # 4) Periodic-background cross-frequency interaction
        # --------------------------------------------------------------
        if self.ablation != 'no_cfi':
            for block in self.interaction_blocks:
                latent = block(latent)

        latent = latent.view(
            batch_size,
            channels,
            self.d_model,
        )
        background_latent, periodic_latent = torch.split(
            latent,
            self.branch_dim,
            dim=-1,
        )

        # --------------------------------------------------------------
        # 5) Frequency-domain prediction
        # --------------------------------------------------------------
        background_features = self.background_predictor(
            background_latent
        )
        periodic_features = self.periodic_residual_predictor(
            periodic_latent
        )

        background_prediction = self._real_features_to_complex(
            background_features
        )
        periodic_correction = self._real_features_to_complex(
            periodic_features
        )

        # The domain-informed 12/24 h prior is used only to guide decomposition of
        # the observed spectrum. The future spectrum is not hard-constrained by
        # a predefined periodic mask; both branches are learned adaptively.
        prediction_spectrum = (
            background_prediction + periodic_correction
        )

        prediction = torch.fft.irfft(
            prediction_spectrum,
            n=self.pred_len,
            dim=-1,
        )

        self.last_diagnostics = {
            "ablation": self.ablation,
            "mask_mode": self.prior.mask_mode,
            "periods": self.prior.period_tensor.detach(),
            "center_bins": self.prior.center_bin_tensor.detach(),
            "input_period_mask": self.prior.input_period_mask.detach(),
            "periodic_input_spectrum": periodic_spectrum.detach(),
            "background_input_spectrum": background_spectrum.detach(),
            "background_prediction_spectrum": background_prediction.detach(),
            "periodic_correction_spectrum": periodic_correction.detach(),
            "prediction_spectrum": prediction_spectrum.detach(),
        }

        return prediction.transpose(1, 2)  # [B, pred_len, N]


class Model(nn.Module):
    """Time-Series-Library compatible forecasting wrapper."""

    def __init__(self, configs):
        super().__init__()

        self.task_name = getattr(
            configs,
            "task_name",
            "long_term_forecast",
        )
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_out = int(getattr(configs, "c_out", 1))

        self.pbi = PBIFrequencyModel(configs)

    def forecast(self, x_enc: torch.Tensor):
        if x_enc.ndim != 3:
            raise ValueError(
                "x_enc must have shape [B, L, N], "
                f"got {tuple(x_enc.shape)}"
            )
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {x_enc.shape[1]}"
            )

        # Same sample-wise normalization/denormalization used by SAP-CFI.
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(
                centered,
                dim=1,
                keepdim=True,
                unbiased=False,
            )
            + 1e-5
        )
        normalized = centered / stdev

        prediction = self.pbi(normalized)
        return prediction * stdev + means

    def get_last_diagnostics(self):
        """Return detached domain-informed masks/spectra from the latest forward."""
        return self.pbi.last_diagnostics

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        if self.task_name not in {
            "long_term_forecast",
            "short_term_forecast",
        }:
            raise NotImplementedError(
                "PBI-Net currently supports forecasting tasks only, got "
                f"{self.task_name!r}"
            )

        return self.forecast(x_enc)
