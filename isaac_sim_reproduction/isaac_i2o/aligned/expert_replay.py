"""Explicit expert provenance, independent of successful-pool membership.

The original GPUReplay remains available for historical controls. This corrected
variant requires provenance at every insertion and preserves it in checkpoints.
"""
import numpy as np
from aligned.gpu_replay import GPUReplay


class ExpertReplay(GPUReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_ids = []

    def add(self, data, successful, path, *, is_expert):
        if not isinstance(is_expert, bool):
            raise TypeError('Explicit boolean expert provenance is required')
        start = self.used
        super().add(data, successful, path)
        self.demo[start:start + len(data['raw_actions'])] = float(is_expert)
        self.episodes[-1]['is_expert'] = is_expert
        if is_expert:
            self.expert_ids.append(len(self.episodes) - 1)

    def sample(self, count=256, demonstrations_only=False):
        if not demonstrations_only:
            return super().sample(count)
        # Base class uses success_ids for this mode. Temporarily provide only
        # proven original experts, never promoted online successes.
        successful = self.success_ids
        self.success_ids = self.expert_ids
        try:
            return super().sample(count, demonstrations_only=True)
        finally:
            self.success_ids = successful

    def files(self):
        return [{k: e[k] for k in ('file', 'successful', 'is_expert')}
                for e in self.episodes]


def migrate_replay_entries(entries, expert_files):
    """Derive old provenance from an allowlist, never from success/demo flags."""
    from pathlib import Path
    allowed = {str(Path(p).resolve()) for p in expert_files}
    result = []
    for entry in entries:
        e = dict(entry)
        expected = str(Path(e['file']).resolve()) in allowed
        if 'is_expert' in e and e['is_expert'] != expected:
            raise ValueError('Expert provenance disagrees with original manifest')
        e['is_expert'] = expected
        result.append(e)
    return result
