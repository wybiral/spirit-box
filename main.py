import argparse
import asyncio
import random
import signal as sig

running = False

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

    def __init__(self, freq=0, rate=1200000, offset=250000):
        self.sdr = RtlSdr()
        self.freq = freq
        self.rate = rate
        self.offset = offset
        self.sdr.gain = 'auto'
        self.sdr.sample_rate = self.rate

    def set_freq(self, freq):
        self.freq = freq
        self.sdr.center_freq = int(freq * 1000000) - self.offset

    async def stream(self, seconds):
        sdr = self.sdr
        rate = self.rate
        n = int(seconds * rate)
        # these SDR modules seem to want kb multiples
        n = n - (n % 1024)
        b = None
        async for x in self.sdr.stream():
            if b is None:
                b = x
            else:
                b = np.concatenate((b, x))
            if len(b) < n:
                continue
            s, b = b[:n], b[n:]
            s = np.array(s)
            yield s.astype('complex64')

    def decode_fm(self, s):
        bandwidth = 200000
        rate = self.rate
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


async def main(loop):
    global running
    parser = argparse.ArgumentParser(description='SDR spirit box.')
    parser.add_argument(
        '--rate',
        type=int,
        default=1200000,
        help='sample rate for SDR device in Hz (default=1200000)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.2,
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
        default=110,
        help='maximum scan frequency (default=109)'
    )
    args = parser.parse_args()
    try:
        import pygame
        from pygame.sndarray import numpysnd
    except ImportError:
        print('pygame not installed, try: pip install pygame')
        exit(1)
    pygame.mixer.init(44100, -16, 1)
    rate = args.rate
    d = args.delay
    f0 = args.min
    f1 = args.max
    radio = Radio(freq=f0, rate=rate)
    running = True
    async for s in radio.stream(args.duration):
        if not running:
            break
        print('frequency=%.02f' % radio.freq)
        s = radio.decode_fm(s)
        sound = numpysnd.make_sound(s)
        sound.play()
        radio.set_freq(f0 + random.random() * (f1 - f0))
        await asyncio.sleep(d)
    await radio.sdr.stop()
    radio.sdr.close()
    pygame.mixer.quit()
    loop.stop()

def handle_sigint(loop, future):
    global running
    running = False
    print('\nStopping...')

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    future = asyncio.gather(main(loop))
    loop.add_signal_handler(sig.SIGINT, handle_sigint, loop, future)
    loop.run_forever()
    print('Done!')