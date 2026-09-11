"""Resolve the reader-facing target-view layout and inference grouping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fdanyone.config import CAMERA
from fdanyone.errors import ConfigurationError

VALID_VIEWS_PER_GROUP = (4, 6)
MIN_PITCH = -15
MAX_PITCH = 45

# RCP uses the canonical proposal cameras seen during training. The resolved
# group size selects a prefix; at most the first four become target references.
RCP_CAMERA_ORDER = (4, 9, 14, 19, 0, 12)

# Second RCP round (paper Sec. 3.3: "Round 2 generates four additional reference
# videos conditioned on the source plus the round-1 references"). They fill the
# gaps left by the source view and round 1. The DiT packs them at 4x4 while
# round 1 keeps the 4x-cheaper 2x2 slots, matching the trained v_src == 8 layout
# in wan_video_dit.py. Without this round the released path only ever builds
# v_src == 5, and viewpack_embedding.proj_4x stays dead despite being trained.
RCP_ROUND2_COUNT = 4


def _farthest_ring_cameras(taken: Sequence[int], count: int) -> tuple[int, ...]:
    """Pick ``count`` canonical-ring cameras farthest from the ones already in use.

    The paper selects reference viewpoints by farthest-point sampling "starting
    from the source view and existing reference views". Distance is circular over
    the canonical ring, and ties break on the lowest index so a run is reproducible.
    """

    ring = CAMERA.count

    def circular(a: int, b: int) -> int:
        delta = abs(a - b) % ring
        return min(delta, ring - delta)

    selected = [index % ring for index in taken]
    chosen: list[int] = []
    for _ in range(count):
        candidates = [index for index in range(ring) if index not in selected]
        if not candidates:
            break

        def rank(index: int) -> tuple[int, int, int]:
            gaps = sorted(circular(index, other) for other in selected)
            # Farthest-point first. Ties are common on a regular ring, so break
            # them toward the widest gap (nearest neighbour on each side), which
            # spreads the round evenly instead of clustering at low indices.
            return (gaps[0], gaps[1] if len(gaps) > 1 else 0, -index)

        pick = max(candidates, key=rank)
        chosen.append(pick)
        selected.append(pick)
    return tuple(sorted(chosen))


@dataclass(frozen=True)
class TargetView:
    """One requested camera in the layer-major public view order."""

    camera_id: int
    layer_index: int
    pitch: int
    yaw: float


@dataclass(frozen=True)
class ViewPlan:
    """Validated target cameras and method components for one run."""

    views_per_layer: int
    layer_pitches: tuple[int, ...]
    start_yaw: int
    yaw_span: int
    views_per_group: int
    enable_rcp: bool
    enable_tcr: bool
    enable_rcp2: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layer_pitches)

    @property
    def num_target_views(self) -> int:
        return self.views_per_layer * self.num_layers

    @property
    def num_groups(self) -> int:
        return self.num_target_views // self.views_per_group

    @property
    def target_views(self) -> tuple[TargetView, ...]:
        step = self.yaw_span / self.views_per_layer
        return tuple(
            TargetView(
                camera_id=layer_index * self.views_per_layer + view_index,
                layer_index=layer_index,
                pitch=pitch,
                yaw=self.start_yaw + view_index * step,
            )
            for layer_index, pitch in enumerate(self.layer_pitches)
            for view_index in range(self.views_per_layer)
        )

    @property
    def front_camera_ids(self) -> tuple[int, ...]:
        return tuple(view.camera_id for view in self.target_views if abs(view.yaw % 360.0) < 1e-8)

    @property
    def rcp_camera_ids(self) -> tuple[int, ...]:
        return RCP_CAMERA_ORDER[: self.views_per_group] if self.enable_rcp else ()

    @property
    def rcp_2x_slot_ids(self) -> tuple[int, ...]:
        """Which round-1 references take the three higher-fidelity 2x2 slots.

        The trained ``v_src == 8`` layout has room for only three of round 1's four
        references, so one is dropped from the target context (it still serves as
        context for round 2). Taking a list prefix drops an arbitrary one: on the
        first 48-view run that dropped yaw 285 and left a 150 degree mid-fidelity
        gap across yaw 210 to 360, which is exactly the arc whose generated output
        changed most and lost its background detail.

        Rule: drop the reference closest to the source view, because the source
        already covers that region at full 1x1 fidelity. That keeps the remaining
        three spread across the hemispheres the source cannot see.
        """

        ids = self.rcp_camera_ids[:4]
        if len(ids) <= 3:
            return ids
        ring = CAMERA.count

        def distance_from_source(camera_id: int) -> int:
            offset = camera_id % ring
            return min(offset, ring - offset)

        nearest = min(ids, key=lambda cid: (distance_from_source(cid), cid))
        return tuple(cid for cid in ids if cid != nearest)

    @property
    def rcp2_camera_ids(self) -> tuple[int, ...]:
        """Round-2 reference cameras; empty unless the second round is enabled.

        Round 1 takes a prefix of RCP_CAMERA_ORDER whose length follows the group
        size, so a fixed round-2 list would collide with it at group size 6.
        Selecting by farthest-point sampling over what round 1 left free keeps the
        two rounds disjoint and evenly spread for every group size.
        """

        if not (self.enable_rcp and self.enable_rcp2):
            return ()
        return _farthest_ring_cameras(
            taken=(0, *self.rcp_camera_ids),  # camera 0 is where the source view sits
            count=RCP_ROUND2_COUNT,
        )

    @property
    def is_canonical_target_ring(self) -> bool:
        return (
            self.views_per_layer == CAMERA.count
            and self.layer_pitches == (int(CAMERA.pitch_degrees),)
            and self.start_yaw == 0
            and self.yaw_span == 360
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "views_per_layer": self.views_per_layer,
            "layer_pitches": list(self.layer_pitches),
            "start_yaw": self.start_yaw,
            "yaw_span": self.yaw_span,
            "views_per_group": self.views_per_group,
            "enable_rcp": self.enable_rcp,
            "enable_tcr": self.enable_tcr,
            "enable_rcp2": self.enable_rcp2,
        }

    @classmethod
    def from_dict(cls, value: object) -> ViewPlan:
        if not isinstance(value, dict):
            raise ConfigurationError("View plan must be a JSON object.")
        try:
            return resolve_view_plan(
                views_per_layer=value["views_per_layer"],
                layer_pitches=value["layer_pitches"],
                start_yaw=value["start_yaw"],
                yaw_span=value["yaw_span"],
                views_per_group=value["views_per_group"],
                enable_rcp=value["enable_rcp"],
                enable_tcr=value["enable_tcr"],
                # Absent in results produced before the second RCP round existed.
                enable_rcp2=bool(value.get("enable_rcp2", False)),
            )
        except KeyError as exc:
            raise ConfigurationError(f"View plan is missing {exc.args[0]!r}.") from None


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer, got {value!r}.")
    return value


def _layer_pitches(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ConfigurationError("layer_pitches must be a non-empty list of integer degrees.")
    pitches = tuple(_integer("Each layer pitch", pitch) for pitch in value)
    if len(set(pitches)) != len(pitches):
        raise ConfigurationError(f"layer_pitches must not contain duplicates, got {list(pitches)}.")
    invalid = [pitch for pitch in pitches if not MIN_PITCH <= pitch <= MAX_PITCH]
    if invalid:
        raise ConfigurationError(
            f"Each layer pitch must be between {MIN_PITCH} and {MAX_PITCH} degrees, got {invalid}."
        )
    return pitches


def _group_size(value: int | str, views_per_layer: int) -> int:
    if isinstance(value, str):
        if value.lower() == "auto":
            divisors = tuple(size for size in VALID_VIEWS_PER_GROUP if views_per_layer % size == 0)
            if not divisors:
                raise ConfigurationError(f"views_per_layer ({views_per_layer}) must be divisible by 4 or 6.")
            return max(divisors)
        try:
            value = int(value)
        except ValueError:
            raise ConfigurationError(
                f"views_per_group must be 'auto' or one of {VALID_VIEWS_PER_GROUP}, got {value!r}."
            ) from None
    value = _integer("views_per_group", value)
    if value not in VALID_VIEWS_PER_GROUP:
        raise ConfigurationError(f"views_per_group must be one of {VALID_VIEWS_PER_GROUP}, got {value!r}.")
    if views_per_layer % value:
        raise ConfigurationError(f"views_per_layer ({views_per_layer}) must be divisible by views_per_group ({value}).")
    return value


def resolve_view_plan(
    *,
    views_per_layer: int = 24,
    layer_pitches: Sequence[int] = (15,),
    start_yaw: int = 0,
    yaw_span: int = 360,
    views_per_group: int | str = "auto",
    enable_rcp: bool = True,
    enable_tcr: bool = True,
    enable_rcp2: bool = False,
) -> ViewPlan:
    """Validate the compact CLI settings before expensive work starts."""

    views_per_layer = _integer("views_per_layer", views_per_layer)
    if views_per_layer <= 0:
        raise ConfigurationError(f"views_per_layer must be positive, got {views_per_layer}.")
    pitches = _layer_pitches(layer_pitches)
    start_yaw = _integer("start_yaw", start_yaw)
    start_yaw = (start_yaw + 180) % 360 - 180
    yaw_span = _integer("yaw_span", yaw_span)
    if not 0 < yaw_span <= 360:
        raise ConfigurationError(f"yaw_span must be between 1 and 360 degrees, got {yaw_span}.")
    resolved_group_size = _group_size(views_per_group, views_per_layer)
    if not isinstance(enable_rcp, bool):
        raise ConfigurationError(f"enable_rcp must be True or False, got {enable_rcp!r}.")
    if not isinstance(enable_tcr, bool):
        raise ConfigurationError(f"enable_tcr must be True or False, got {enable_tcr!r}.")
    if not isinstance(enable_rcp2, bool):
        raise ConfigurationError(f"enable_rcp2 must be True or False, got {enable_rcp2!r}.")

    # Up to six requested targets are cheaper and clearer to generate directly.
    rcp_active = enable_rcp and views_per_layer * len(pitches) > 6
    # The trained v_src == 8 packing needs four round-1 references in hand, so
    # the second round is only meaningful once round 1 produces at least four.
    if enable_rcp2 and rcp_active and resolved_group_size < 4:
        raise ConfigurationError("enable_rcp2 needs at least four round-1 references (views_per_group >= 4).")
    return ViewPlan(
        views_per_layer=views_per_layer,
        layer_pitches=pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        views_per_group=resolved_group_size,
        enable_rcp=rcp_active,
        enable_tcr=enable_tcr,
        enable_rcp2=bool(enable_rcp2 and rcp_active),
    )
