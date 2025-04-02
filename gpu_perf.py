class GPU_perf():
    def __init__(self,sm,comm_sm, fp16_flops,fp8_flops,mem,mem_bw, nvlink_bw,pcie_bw, discount_rate):
        self.sm = sm
        self.comm_sm = comm_sm #用于通信的SM数量
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate #整体性能按峰值性能折扣

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate  * ( self.sm  - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw *  self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw *  self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw *  self.discount_rate