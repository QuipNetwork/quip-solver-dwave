"""Build the engine's programs file for the pipeline from a calibrated screen."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

REHEAT_BETA = 0.25
GPU_DEEP_SWEEPS = 1_048_576  # the depth of the deep leg of the GPU arm, the same as ``cold-1048576`` of the QPU arms
OBSERVE_EVERY = 16_384  # the observe interval of the deep programs of a run
MAX_LEGS = 16  # the most legs of one program (the engine's ``MAX_SEGS``)


def beta_tag(beta: float) -> str:
    """A name part for a start beta: 0.25 is ``b0250``."""
    return f"b{round(beta * 1000):04d}"


def cold_name(sweeps: int) -> str:
    return f"cold-{sweeps}"


def seeded_name(sweeps: int, beta: float) -> str:
    return f"seeded-{sweeps}-{beta_tag(beta)}"


def chain_name(legs: int, sweeps: int, beta: float) -> str:
    return f"chain-{legs}x{sweeps}-{beta_tag(beta)}"


def chain_program(
    legs: int, sweeps: int, beta: float, words: int = 2, observe_every: int = OBSERVE_EVERY
) -> Dict[str, Any]:
    """A deep program of ``legs`` legs of ``sweeps`` sweeps, each from ``beta`` to the cold end.

    The first leg starts from the lanes of the answer, and each later leg reheats the lanes that the leg before it
    left. The miner runs one leg at a time and pushes the best reads so far once: when the target admits them, or after
    the last leg. On the 96 newest winners, 16 legs of 131,072 sweeps from QPU reads won more rounds than one leg of
    the same sweeps, or a ladder of legs that double.
    """
    if not 1 <= legs <= MAX_LEGS:
        raise ValueError(f"legs must be 1 to {MAX_LEGS}")
    leg = {"from_beta": beta, "sweeps": sweeps}
    return {"name": chain_name(legs, sweeps, beta), "words": words, "observe_every": observe_every,
            "legs": [{"start": "seeded", **leg}] + [{"start": "continue", **leg} for _ in range(legs - 1)]}


def gpu_256_program(
    leg256: Dict[str, Any], deep_sweeps: int = GPU_DEEP_SWEEPS, observe_every: int = OBSERVE_EVERY
) -> Dict[str, Any]:
    """The program of the GPU arm: the legs of ``leg-256`` with their gates, and then a fresh deep leg.

    A survivor that passes the gate of the last leg starts the deep leg at once, on the GPU alone. The miner runs this
    program when it has no sidecar, and the sidecar runs it for a survivor that stays off the QPU, so a programs file
    without it makes both refuse to start.
    """
    return {"name": "gpu-256", "words": leg256["words"], "observe_every": observe_every,
            "legs": [dict(leg) for leg in leg256["legs"]] + [{"start": "fresh", "sweeps": deep_sweeps}]}


def pipeline_programs(
    screen: Dict[str, Any],
    *,
    handoff_legs: int,
    handoff_margin: int,
    depths: Sequence[int],
    start_betas: Sequence[float],
    observe_every: int,
) -> List[Dict[str, Any]]:
    """Programs for one pipeline run.

    ``screen`` is the gated screen program that ``cascade_gates.py msa`` wrote.
    The pipeline screens through its first ``handoff_legs`` legs and loosens the
    last of those gates by ``handoff_margin`` energy units, so more models reach
    the QPU. Each later screen leg becomes a ``leg-<n>`` program that continues
    from a survivor's own states, with its calibrated gate. Each depth gets one
    cold program and one seeded program for each start beta. The program of the
    GPU arm, ``gpu-256``, comes last when the screen has a ``leg-256``.
    """
    legs = screen["legs"]
    if not 1 <= handoff_legs <= len(legs):
        raise ValueError(f"handoff_legs must be 1 to {len(legs)}")
    words = screen["words"]
    head = [dict(leg) for leg in legs[:handoff_legs]]
    head[-1]["gate"] = head[-1]["gate"] + handoff_margin
    programs: List[Dict[str, Any]] = [{"name": "screen", "words": words, "observe_every": 0, "legs": head}]
    done = sum(leg["sweeps"] for leg in head)
    for leg in legs[handoff_legs:]:
        done += leg["sweeps"]
        programs.append(
            {"name": f"leg-{done}", "words": words, "observe_every": 0,
             "legs": [{"start": "seeded", "from_beta": leg.get("from_beta", REHEAT_BETA), "sweeps": leg["sweeps"],
                       "gate": leg["gate"]}]}
        )
    for sweeps in depths:
        programs.append({"name": cold_name(sweeps), "words": words, "observe_every": observe_every,
                         "legs": [{"start": "fresh", "sweeps": sweeps}]})
        for beta in start_betas:
            programs.append({"name": seeded_name(sweeps, beta), "words": words, "observe_every": observe_every,
                             "legs": [{"start": "seeded", "from_beta": beta, "sweeps": sweeps}]})
    leg256 = next((p for p in programs if p["name"] == "leg-256"), None)
    if leg256 is not None:
        programs.append(gpu_256_program(leg256, observe_every=observe_every))
    return programs


def start_beta(hot: float, cold: float, fraction: float) -> float:
    """The beta at ``fraction`` of the log-spaced ladder, as ``seeded_sweep.py`` defines it."""
    return hot * (cold / hot) ** fraction
