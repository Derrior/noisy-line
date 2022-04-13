import matplotlib.pyplot as plt
from math import pi

class PeakDetectionMethod:
    def detect_peaks(self, point_list, max_amount):
        pass


class NausWallenstein(PeakDetectionMethod):
    
    def binomial(self, k, M, p):
        cached = 0
        cached = self.binomial_cache.get((k, M, p), 0)
        if cached != 0:
            return cached
        cached = self.binomial_cache.get((k - 1, M, p), 0)
        result = 1
        if cached == 0:
            result *= p ** k
            result *= (1 - p) ** (M - k)

            for i in range(M - k + 1, M + 1):
                result *= i
            for i in range(1, k + 1):
                result /= i
        else:
            result = cached * p / (1 - p)
            result *= M - k + 1
            result /= k
        self.binomial_cache[(k, M, p)] = result
            
        return result

    def __init__(self, alpha=0.01, r_window=50, phi_window=0.05):
        self.r_window = r_window
        self.phi_window = phi_window
        self.alpha = alpha
        self.result = {}
        self.binomial_cache = {}
        self.binomial_sums = {}
        self.outer_windows = {}
        
    def count_statistic(self, observed, expected, window, time, total):
        if observed <= expected:
            return 1
        binomial_params = (total, int(expected * time / window), window / time)
        
        binomial_sums = self.binomial_sums.get(
                            binomial_params,
                            None)
        if binomial_sums == None:
            binomial_sums = [self.binomial(i, *binomial_params[1:]) for i in range(0, total + 1)]
            for i in range(total, 0, -1):
                binomial_sums[i - 1] += binomial_sums[i]
            binomial_sums.append(0)
            self.binomial_sums[binomial_params] = binomial_sums
        if observed > total:
            print("wrong")
        elif observed == total:
            print("total")
        ret = ((observed - expected) * (time / window) + 1) * self.binomial(observed, *binomial_params[1:]) + \
            2 * binomial_sums[observed + 1]
        if ret < 0:
            print(ret)
        return ret
    
    def window_search(self, point_list, time, key, functor, window=None):
        points_sorted = sorted(point_list, key=key)
        if window is None:
            window = time / 100
        expected = len(points_sorted) * window / time
        begin = 0
        end = 1
        while end < len(points_sorted):
            while end < len(points_sorted) and key(points_sorted[end]) - key(points_sorted[begin]) <= window:
                end += 1
            observed = end - begin
            center = sum(map(key, points_sorted[begin:end])) / (end - begin)
            functor(points_sorted[begin:end], center,
                window, time, len(points_sorted))
            if end != len(points_sorted):
                while begin < len(points_sorted) and key(points_sorted[end]) - key(points_sorted[begin]) > window:  
                    begin += 1
    
    def add_new_statistic(self, y_center, points, x_center, w, T, N):
        stat = self.count_statistic(len(points), N * w / T, w, T, N)
        if stat < self.alpha:
            self.result[(x_center, y_center)] = stat
    
    def inner_search(self, point_list, center, w, T, N):
        observed = len(point_list)
        expected = N * w / T
        stat = self.count_statistic(len(point_list), N * w / T, w, T, N)

        self.outer_windows[center] = (len(point_list), stat)
        if stat > self.alpha:
            return
        x_key = lambda x: x[0]
        functor = lambda *x: self.add_new_statistic(center, *x)
        self.window_search(point_list, max(point_list, key=x_key)[0], x_key, functor, window=self.r_window)
        
    def detect_peaks(self, point_list, max_amount):
        self.result = {}
        #x_key = lambda x: x[0]
        y_key = lambda y: y[1]
        functor = self.inner_search
        time = y_key(max(point_list, key=y_key))\
                - y_key(min(point_list, key=y_key))
        self.window_search(point_list, 2 * pi, y_key, functor, window=self.phi_window)
        return self.result, self.outer_windows
        """
        points_sorted_x = sorted(point_list, key=lambda x: x[0])
        points_sorted_y = sorted(point_list, key=lambda x: x[1])
        
        x_limits = points_sorted_x[0][0], points_sorted_x[-1][0]
        x_time = (x_limits[1] - x_limits[0])
        window = time / 40
        expected = len(points_sorted_x) * window / time
        begin = 0
        end = 1
        while end < len(points_sorted_x): # window contains points with indexes [begin, end)
            while end < len(points_sorted_x) and points_sorted_x[end][0] - points_sorted_x[begin][0] < window:
                end += 1
            observed = end - begin
            if observed > expected:
                print(observed, expected, window, time, len(points_sorted_x))
                current_points = copy.copy(points_sorted_x[begin:end])
                current_points.sort(key=lambda x: x[1])
                
                
                stat = self.count_statistic(observed, expected, window, time, len(points_sorted_x))
                print(stat, points_sorted_x[begin][0], points_sorted_x[end - 1][0])

            if end != len(points_sorted_x):
                while begin < len(points_sorted_x) and points_sorted_x[end][0] - points_sorted_x[begin][0] > window:  
                    begin += 1
                    """
           
