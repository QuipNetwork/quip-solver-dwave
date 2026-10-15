#!/usr/bin/env python3
"""Check, on real hardware, that SAPI's timestamps reach SampleResult.

One job of two qubits, one read: about 15 ms of QPU access time. Prints
the submit-to-solve time the server reports, the access time it bills, and
the queue wait that is their difference. Exits 1 when the timestamps are
missing, because the history's D-Wave-side split depends on them.
"""

import sys
import time

import numpy as np

from quip_solver_dwave.ocean import OceanSampler


def main() -> int:
    sampler = OceanSampler(mock=False)
    sampler.ensure_connected()
    u, v = sampler.live_edges[0]
    t0 = time.perf_counter()
    result = sampler.sample(
        np.array([u, v]),
        np.zeros(2),
        np.array([[u, v]]),
        np.array([-1.0]),
        num_reads=1,
        label="quip-probe-timestamps",
    )
    rtt = time.perf_counter() - t0
    try:
        print(f"round trip     : {rtt * 1000:8.1f} ms")
        print(f"access time    : {result.device_access_time_us / 1000:8.1f} ms")
        print(f"in flight      : {result.inflight_at_submit}")
        if result.submitted_on_s is None or result.solved_on_s is None:
            print("FAIL: submitted_on/solved_on missing from the SAPI message")
            return 1
        sapi = result.solved_on_s - result.submitted_on_s
        access = result.device_access_time_us / 1_000_000
        print(f"SAPI service   : {sapi * 1000:8.1f} ms (solved_on - submitted_on)")
        print(f"queue wait     : {(sapi - access) * 1000:8.1f} ms")
        print(f"client+network : {(rtt - sapi) * 1000:8.1f} ms")
        print(f"extra          : {result.extra}")
        return 0
    finally:
        sampler.close()


if __name__ == "__main__":
    sys.exit(main())
