import numpy as np


def downsample(x, decimation, phase=0):
    """

    :param x:
    :param decimation:
    :param phase
    :return:
    """
    return x[phase::decimation]


def upsample(x, rate):
    """

    :param x:
    :param rate:
    :return:
    """
    output = np.zeros(len(x), x.dtype)
    output[::rate] = x
    return output


def rrcosfilt(samples_per_symbol, span, beta, Ts=1):
    """

    :param samples_per_symbol:
    :param span:
    :param beta:
    :param Ts:
    :return:
    """
    t = np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
    output = np.zeros(len(t))
    idx0 = (t == 0)
    idx1 = (np.abs(t) == 1 / (4 * beta))
    output[idx0] = (1 + beta * (4 / np.pi - 1)) / Ts
    output[idx1] = (beta / np.sqrt(2) * (
            (1 + 2 / np.pi) * np.sin(np.pi / 4 / beta) + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta))) / Ts
    t = t[~(idx0 | idx1)]
    output[~(idx0 | idx1)] = (np.sin(np.pi * t / Ts * (1 - beta)) + 4 * beta * t / Ts * np.cos(
        np.pi * t / Ts * (1 + beta))) / (Ts * np.pi * t / Ts * (1 - (4 * beta * t / Ts) ** 2))
    return output


def tobits(s):
    """

    :param s:
    :return:
    """
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    """

    :param bits:
    :return:
    """
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def modulate(inphase, quadrature, normalized_freq):
    """

    :param inphase:
    :param quadrature:
    :param normalized_freq:
    :return:
    """
    arg = 2 * np.pi * normalized_freq * np.arange(len(inphase))
    output = (inphase * np.cos(arg) - quadrature * np.sin(arg)) * np.sqrt(2)
    return output


def demodulate(data, normalized_freq):
    """

    :param data:
    :param normalized_freq:
    :return:
    """
    arg = 2 * np.pi * normalized_freq * np.arange(len(data))
    in_phase = data * np.cos(arg) * np.sqrt(2)
    quadrature = -data * np.sin(arg) * np.sqrt(2)

def matched_filter(data, pulse_shape):
    return np.convolve(data, pulse_shape)


def mqam_encode(data, m):
    n = np.log2(m)
    

def encode(data, modulation_dict):


if __name__ == '__main__':
    pulse_shape = rrcosfilt(8, 12, .5)
