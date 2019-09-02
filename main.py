import argparse
import random

try:
    import numpy as np
except ImportError:
    print('numpy not installed, try: pip install numpy')
    exit(1)

try:
    from rtlsdr import RtlSdr
except ImportError:
    print('pyrtlsdr not installed, try: pip install pyrtlsdr')
    exit(1)

try:
    import scipy.signal as signal
except ImportError:
    print('scipy not installed, try: pip install scipy')
    exit(1)


class Radio:

    def __init__(self):
        self.sdr = RtlSdr()
        self.offset = 250000
        self.rate = 1140000
        self.sdr.gain = 'auto'
        self.sdr.sample_rate = self.rate

    def capture_fm(self, freq, seconds):
        bandwidth = 200000
        rate = self.rate
        n = int(seconds * rate)
        n = n - (n % 1024)
        f = int(freq * 1.0e6)
        self.sdr.center_freq = f - self.offset
        s = self.sdr.read_samples(n)
        s = np.array(s).astype("complex64")
        # shift signal
        f = -1.0j * 2.0 * np.pi * self.offset / rate
        s *= np.exp(f * np.arange(len(s)))
        # downsample to FM bandwidth
        d = int(rate / bandwidth)  
        s = signal.decimate(s, d)  
        rate2 = rate / d  
        # polar discriminator
        s = np.angle(s[1:] * np.conj(s[:-1]))
        # de-emphasis filter
        x = np.exp(-1 / (rate2 * 75e-6))
        s = signal.lfilter([1 - x], [1, -x], s)
        # resample to 44100
        s = signal.decimate(s, int(rate2 / 44100))
        # scale volume
        s *= 10000 / np.max(np.abs(s))  
        return s.astype("int16")


def main():
    parser = argparse.ArgumentParser(description='SDR spirit box.')
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='delay between scans in seconds (default=0)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=0.3,
        help='duration of each scan in seconds (default=0.3)'
    )
    parser.add_argument(
        '--min',
        type=float,
        default=87,
        help='minimum scan frequency (default=87)'
    )
    parser.add_argument(
        '--max',
        type=float,
        default=109,
        help='maximum scan frequency (default=109)'
    )
    args = parser.parse_args()
    try:
        import pygame
    except ImportError:
        print('pygame not installed, try: pip install pygame')
        exit(1)
    pygame.mixer.init(44100, -16, 1)
    radio = Radio()
    delay = args.delay
    duration = args.duration
    min_freq = args.min
    max_freq = args.max
    try:
        while True:
            freq = min_freq + random.random() * (max_freq - min_freq)
            freq = round(freq * 10) / 10
            s = radio.capture_fm(freq, duration)
            sound = pygame.sndarray.numpysnd.make_sound(s)
            sound.play()
            pygame.time.wait(int(delay * 1000))
    except KeyboardInterrupt:
        print('')
    except Exception as e:
        print(e)
    radio.sdr.close()  
    pygame.mixer.quit()

if __name__ == '__main__':
    main()