import libdp

kDprofile_Metric_DramRead = 0
kDprofile_Metric_DramWrite = 1
kDprofile_Metric_L1Load = 2
kDprofile_Metric_L1Store = 3
kDprofile_Metric_L2Load = 4
kDprofile_Metric_L2Store = 5

class DProfile:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DProfile, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._profiler = libdp.DProfile()

    def start_profile(self, metric_set:set[str]):
        self._profiler.start_profile(metric_set)
        
    def stop_profile(self) -> dict[str:list[float]]:
        return self._profiler.stop_profile()


__all__ = [ 
    'DProfile',
    'kDprofile_Metric_DramRead',
    'kDprofile_Metric_DramWrite',
    'kDprofile_Metric_L1Load',
    'kDprofile_Metric_L1Store',
    'kDprofile_Metric_L2Load',
    'kDprofile_Metric_L2Store'
]
