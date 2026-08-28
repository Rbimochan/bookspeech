"""Rough conversion-time estimate surfaced to the user before starting a job.

The constant below is seeded from the benchmark script (scripts/benchmark_synthesis.py)
run on this machine's CPU; re-run it and update this value if hardware changes
meaningfully, or wire it to persisted benchmark results for real accuracy.
"""

# Characters of source text Kokoro synthesizes per second, CPU inference.
# Conservative default — actual throughput varies with hardware and provider
# (GPU/CoreML will be faster than this).
DEFAULT_CHARS_PER_SEC = 100.0

# Assembly (concat + AAC encode) is fast relative to synthesis but not free.
ASSEMBLY_OVERHEAD_FRACTION = 0.05


def estimate_conversion_seconds(total_chars: int, chars_per_sec: float = DEFAULT_CHARS_PER_SEC) -> float:
    synthesis_seconds = total_chars / chars_per_sec
    return synthesis_seconds * (1 + ASSEMBLY_OVERHEAD_FRACTION)
