from dataclasses import dataclass, asdict, field
from typing import List, Tuple


@dataclass
class ROIData:
    amyg_vox: List
    h_range: List
    w_range: List
    d_range: List


@dataclass
class SubjectMetaData:
    subject_name: str
    watch_on: List[int]
    watch_duration: List[int]
    regulate_on: List[int]
    regulate_duration: List[int]
    initial_delay: int = 2
    subject_type: str = 'healthy'

    def gen_time_range(self, on, duration): return list(range(int(on + self.initial_delay), int(on + duration)))

    def __post_init__(self):
        self.min_w = min(self.watch_duration + self.regulate_duration) - self.initial_delay
        self.watch_times = map(self.gen_time_range, self.watch_on, self.watch_duration)
        self.regulate_times = map(self.gen_time_range, self.regulate_on, self.regulate_duration)
