import time

class TerminalLogger():
    
    def __init__(self, max_steps):
        self._tmp_dict = {}    
        self._tmp_val_dict = {}
        self.n = 0
        self.start_time = time.time()
        self.max_steps = max_steps
        self.last_step = 0
        pass
    
    def log(self, name, value, sync_dict=False):
        if sync_dict:
            self._tmp_val_dict[name] = value + self._tmp_val_dict.get(name, 0)
            self.n = self.n + 1
        else:
            self._tmp_dict[name] = value
        pass
    
    def log_dict(self, data, sync_dict=False):
        for k, v in data.items():
            self.log(k, v, sync_dict=sync_dict)
        pass
    

    def print(self, print_fn, step, valid_mode=False):
        current_time = time.time()
        if valid_mode:
            print_str = "| valid | step: {:>5}/{:<5} |".format(step, self.max_steps)
            for k, v in self._tmp_val_dict.items():
                print_str += " {:>5}: {:>10.6f} |".format(k, v / self.n)
            print_fn(print_str)
            self._tmp_val_dict = {}
            self.n = 0
        else:
            print_str = "| train | step: {:>5}/{:<5} |".format(step, self.max_steps)
            remaining_time = self.estimate_remaining_time(self.max_steps, step, current_time, self.start_time)
            print_str += " time: {:>10} |".format(remaining_time)
            for k, v in self._tmp_dict.items():
                print_str += " {:>5}: {:>10.6f} |".format(k, v)
            print_fn(print_str)
            self._tmp_dict = {}
        # self.start_time = current_time
        # self.last_step = step
        pass

    def estimate_remaining_time(self, total_steps, current_step, current_time, start_time):
        """
        计算预计剩余时间。

        Args:
            total_steps (int): 总的训练步数。
            current_step (int): 当前训练步数。
            start_time (float): 训练开始时间，通过 time.time() 获取。

        Returns:
            str: 预计剩余时间的字符串表示，格式为 "HH:MM:SS"。
        """
        if current_step == 0:
            return "N/A"
        elapsed_time = current_time - start_time
        remaining_steps = total_steps - current_step
        estimated_remaining_time = (elapsed_time / (current_step)) * remaining_steps
        return time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))

    pass