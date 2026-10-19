import matplotlib.pyplot as plt
import numpy as np
import math

# extract HW info specs:
#   lscpu | grep -E 'Model name|Socket|Core|Thread|MHz'
# Get FlopsPerCycle supported tech.: 
#   cat /proc/cpuinfo | grep 'flags' | grep -o 'avx[^ ]*'
# Measured BW via:
#   wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
#   gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream
#   ./stream
# or with likwid for e.g. multiple NUMA Nodes

class Hardware:
    # Theoretical HW Peak Performance, HW Bandwidth
    def __init__(self, PP = None, BW = None):
        self.PP = PP
        self.BW = BW

    # Hardware (theoratical) peak performance (Rpeak), FLOPs/sec
    def setPP(self, ClockFrequency, FlopsPerCycle, CoresPerCPU, CPUs):
        self.PP = ClockFrequency*FlopsPerCycle*CoresPerCPU*CPUs

    # Hardware bandwidth from measurement of data and time, Bytes/sec
    def setBW(self, Read, Write, Time):
        self.BW = (Read + Write)/Time

    def printAvailableInfo(self):
        if self.PP != None: print(f"Theoretical Peak Performance = {self.PP/10**9:.1f}  GFLOPs/sec")
        if self.BW != None: print(f"Theoretical Max. Bandwidth = {self.BW/10**9:.1f}  GByte/sec")

    # Plot roofline of Roofline-Plot, since this is only HW dependent 
    def plotRoof(self, ax, color, label=None, linestyle="-"):
        def roofline(OI):
            MemoryPerformance = OI * self.BW # Memory Roof
            return MemoryPerformance if MemoryPerformance < self.PP else np.nan
        OIs = np.logspace(np.log10(0.2), np.log10(80000), 100000)
        roof = [roofline(x) for x in OIs]
        if label == None:
            ax.loglog(OIs, roof, c=color, linestyle=linestyle)
        else:
            ax.loglog(OIs, roof, c=color, linestyle=linestyle, label=label)

    def plotCeiling(self, ax, color, label=None, linestyle="-"):
        def roofline(OI):
            MemoryPerformance = OI * self.BW # Memory Roof
            return np.nan if MemoryPerformance < self.PP else self.PP
        OIs = np.logspace(np.log10(0.2), np.log10(80000), 100000)
        roof = [roofline(x) for x in OIs]
        if label == None:
            ax.loglog(OIs, roof, c=color, linestyle=linestyle)
        else:
            ax.loglog(OIs, roof, c=color, linestyle=linestyle, label=label)
    

