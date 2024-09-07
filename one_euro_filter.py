import math
"""
Modify from https://github.com/jaantollander/OneEuroFilter/blob/master/python/one_euro_filter.py
Removed timestamp(t0 and t) as timestamp should be continuous
That is, set t_e to always be 1

'''Bo Chen:
    min_cutoff : control jitter ,if value set to high ,jitter will become serious,delay will be sloved
    beta : control delay , if value set to high , jitter will become serious,delay will be sloved
    need to set min_cutoff first and set beta value become zero ,let jitter become less serious,then
    set beta ,to find the balance.
'''

```ChatGPT:
* t0: Initial timestamp (float) - This parameter represents the timestamp of the first data point. It is used to initialize the filter's internal state.

* x0: Initial data value (float) - This parameter represents the value of the first data point. It is used to initialize the filter's internal state.

* dx0: Initial derivative value (float, optional) - This is the initial derivative value of the data. It is set to 0.0 by default but can be specified if you have prior knowledge of the initial rate of change.

* min_cutoff: Minimum cutoff frequency (float) - This parameter controls the minimum cutoff frequency for filtering. It influences how quickly the filter responds to changes in the input data. Smaller values make the filter more sluggish, while larger values make it more responsive.

* beta: Beta coefficient (float) - This parameter controls the adaptation of the cutoff frequency based on the rate of change of the input data. It determines how much the cutoff frequency can change. A higher beta makes the cutoff frequency adapt more quickly to changes in the input data.

* d_cutoff: Derivative cutoff frequency (float) - This parameter influences the filtering of the derivative of the input data. It helps to filter out noise in the rate of change of the data.
```

```from ChatGPT, about jittering:
* min_cutoff (Minimum Cutoff Frequency):

Increasing min_cutoff can reduce jittering in the filtered result. This parameter sets a lower bound on the cutoff frequency of the filter. A higher min_cutoff value makes the filter less responsive to rapid changes in the input data, effectively smoothing out jitter.
However, setting min_cutoff too high may make the filter excessively sluggish, causing it to respond slowly to genuine changes in the input signal.

* beta (Beta Coefficient):

Increasing beta can help reduce jittering by allowing the filter to adapt more quickly to changes in the input data. This parameter controls the rate at which the cutoff frequency adjusts based on the rate of change of the data.
A higher beta value makes the filter more responsive to sudden changes in the input signal, which can help mitigate jitter.
However, setting beta too high can also make the filter overly sensitive to noise, potentially amplifying small fluctuations in the input data.

* d_cutoff (Derivative Cutoff Frequency):

Adjusting d_cutoff can influence how the filter handles jitter in the derivative (rate of change) of the input data. By filtering the derivative, the filter can attenuate high-frequency noise.
Increasing d_cutoff can reduce jitter in the filtered derivative, which can help smooth out variations in the input signal.
Similar to the other parameters, setting d_cutoff too high can slow down the filter's response to genuine changes in the rate of change of the input data.
```
"""

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, x0, dx0=0.0, min_cutoff=0.1, beta=0.0,
                 d_cutoff=0.4):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)

    def __call__(self, x):
        """Compute the filtered signal."""
        t_e = 1

        # The filtered derivative of the signal.
        # a_d is about 0.8627 when d_cutoff is 1.0
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat