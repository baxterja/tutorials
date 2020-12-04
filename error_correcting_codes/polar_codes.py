import numpy as np
import time

"""
This program is based off the algorithm developed at: https://www.polarcodes.com/
"""


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames
def construct_code(N, kk, p, channel_type='BSC'):
    """
    Finds the bit locations that will have the highest channel capacity
    :param N: integer (power of two)
        The number of physical bits to be sent over a channel at a time (block length)
    :param kk: integer (less than N)
        The number of information bits to send in a block
    :param p: The Design parameter
        p for BSC or eb/n0 for AWGN
    :param channel_type: 'AWGN' or 'BSC'
        The type of channel.
    :return information_indices: (kk,) ndarray of integers
        The high channel capacity bit locations
        Note: everything is done in log space to help with underflow
    """
    from numpy import log
    nn = int(np.log2(N))
    if channel_type == 'BSC':
        start_val = log(2 * np.sqrt(p * (1 - p)))
    else:
        start_val = -(kk / N) * p
    cur_list = np.array([start_val])
    for n in range(nn):
        next_list = np.zeros(2 * len(cur_list))
        offset = 2 ** n
        for i, val in enumerate(cur_list):
            next_list[i] = log(2) + val + np.log1p(-np.exp(2 * val - (log(2) + val)))
            next_list[offset + i] = val * 2
        cur_list = next_list.copy()
    idx = np.argsort(cur_list).astype(np.int)
    information_indices = idx[:kk]

    LUTs = np.zeros((N, 3), np.int)
    for i in range(N):
        i_bin = '{0:0{1}b}'.format(i, nn)[::-1]
        LUTs[i][0] = int(i_bin, 2)
        LUTs[i][1] = len(i_bin) - len(i_bin.lstrip('1')) + 1
        LUTs[i][2] = len(i_bin) - len(i_bin.lstrip('0')) + 1

    return information_indices, LUTs


def embed(data, nn, data_indices):
    """
    Embeds the information bits into the transmission block
    :param data: (k,) array-like of binary characters
        The information data to be sent
    :param nn: int
        log2(block_length)
    :param data_indices: (k,) array-like of int
        List containing high capacity bit locations
    :return:
    """
    d = np.zeros(2 ** nn, dtype=np.int)
    d[data_indices] = data
    return d


def encode(data, nn, data_indices):
    """
    Encodes data at data_indices location
    Effectively performs F @ data, but uses efficient butterfly circuit
    :param data: (kk,) array-like binary characters
        Data to be encoded
    :param nn: int
        log2(block_length)
    :param data_indices: (kk,) array-like integers
        The high channel capacity bit locations
    :return encoded_data: (2**nn,) ndarray
        The encoded data
    """
    d = embed(data, nn, data_indices)
    for i in range(nn):
        B = 2 ** (nn - i)
        nB = 2 ** i
        for j in range(nB):
            base = j * B
            B2 = B // 2
            for ll in range(B2):
                d[base + ll] = d[base + ll] + d[base + B2 + ll]
    return np.mod(d, 2)


# noinspection PyShadowingNames,PyShadowingNames
def get_initial_llrs(data, parameter, channel_type='BSC', k=None, N=None):
    """

    :param data: (N,) ndarray
        The received data (matched filter out.  Soft or Hard depending on channel type)
    :param parameter: The Design parameter
        p for BSC or eb/n0 for AWGN
    :param channel_type: 'AWGN' or 'BSC'
        The type of channel.
    :param k: int
        Only required for 'AWGN' channel (the number of information bits)
    :param N: int
        Only required for 'AWGN' channel (the number of physical bits)
    :return: ()
    """
    if channel_type == 'BSC':
        llr1 = np.log(parameter) - np.log(1 - parameter)
        return -(-1) ** data * llr1
    else:
        if (k is None) or (N is None):
            print('No K or N given, assuming rate 1/2 code')
            k = 2
            N = 1
        return - 2 * np.sqrt(2 * (k / N) * parameter) * data


def func_g(decision, llr1, llr2):
    """
    Function for updating likelihoods if bit was just set
    :param decision: binary digit
        The decision of the bit that was just sent
    :param llr1: float
        The likelihood of the upper branch
    :param llr2: float
        The likelihood of the lower branch.
    :return: The updated likelihood
    """
    return llr2 + (-1) ** decision * llr1


def func_f(llr1, llr2):
    """
    The output likelihood of combination of two likelihoods
    :param llr1: float
        The upper likelihood
    :param llr2: float
        The lower likelihood
    :return: float
        The updated likelihood
    """
    return np.logaddexp(llr1 + llr2, 0) - np.logaddexp(llr1, llr2)


def update_llr(last_level, nn, llr, bits):
    """
    Update the log likelihood ration table
    :param last_level: int
        The last level to update (based off bit reversal)
    :param nn: int
        Number of physical bits
    :param llr: (2*N-1,) ndarray
        the log likelihood ratio table
    :param bits: (2,N-1) ndarray
        the list containing bits decided up to this point
    :return: nothing,however llr will have been updated
    """
    if last_level == (nn + 1):
        next_level = nn
    else:
        start = 2 ** (last_level - 1)
        end = 2 ** last_level - 1
        for idx in range(start, end + 1):
            llr[idx - 1] = func_g(bits[0, idx - 1],
                                  llr[end + 2 * (idx - start)],
                                  llr[end + 2 * (idx - start) + 1])
        next_level = last_level - 1

    for lev in range(next_level, 0, -1):
        start = 2 ** (lev - 1)
        end = 2 ** lev - 1
        idx = np.arange(start, end + 1)
        llr[idx - 1] = func_f(llr[end + 2 * (idx - start)],
                              llr[end + 2 * (idx - start) + 1])


def update_bits(latest_bit, i, last_level, nn, bits):
    """
    Updates the bits table
    :param latest_bit: binary digit (int)
        The hard decision of the latest bit
    :param i: int
        The location of the most recent bit
    :param last_level: int
        The how far back in the tree to update the bit table
    :param nn: int
        the number of physical bits
    :param bits: (2, N-1) ndarray
        The tree containing bit information
    :return:
        Doesn't return anything, but does update the bit tree
    """
    if i < 2 ** nn / 2:
        bits[0, 0] = latest_bit
        return
    if i == 2 ** nn - 1:
        return

    bits[1, 0] = latest_bit
    for lev in range(1, last_level):
        first_idx = int(lev < (last_level - 1))

        start = 2 ** (lev - 1)
        end = 2 ** lev - 1
        idx = np.arange(start, end + 1)
        bits[first_idx, end + 2 * (idx - start)] = np.mod(bits[0, idx - 1] + bits[1, idx - 1], 2)
        bits[first_idx, end + 2 * (idx - start) + 1] = bits[1, idx - 1]


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames
def decode(data, p, information_indices, LUTs, channel_type='BSC'):
    """
    Decodes received data
    :param data: (N,) ndarray
        The received data (hard or soft depending on channel type)
    :param p: float
        The Design parameter, p for BSC or eb/n0 for awgn
    :param information_indices: (k,) ndarray
        The location of the information bits
    :param LUTs: (N,3) ndarray
        Precomputed look up tables for the bit reversed number, first zero and first one variables
    :param channel_type: string
        'BSC' or 'AWGN'
    :return:
    """
    N = len(data)
    nn = int(np.log2(N))
    k = len(information_indices)
    initial_llrs = get_initial_llrs(data, p, channel_type, k, N)
    llr = np.zeros(2 * N - 1)
    llr[N - 1:] = initial_llrs
    bits = np.zeros((2, N - 1))
    d_hat = np.zeros(N)

    for j in range(N):
        LUT = LUTs[j, :]
        i_dec = LUT[0]
        update_llr(LUT[2], nn, llr, bits)
        if i_dec in information_indices:
            d_hat[i_dec] = llr[0] < 0
        else:
            d_hat[i_dec] = 0
        update_bits(d_hat[i_dec], i_dec, LUT[1], nn, bits)

    u = d_hat[information_indices]
    return u


if __name__ == '__main__':
    n = 14
    k = 2 ** (n - 1)
    p = .01
    information_locations, LUTs = construct_code(2 ** n, k, p)
    embedded_data = np.random.randint(0, 2, k)

    start_time = time.time()
    encoded_data = encode(embedded_data, n, information_locations)
    print(time.time() - start_time)
    error = np.random.binomial(1, p, len(encoded_data))
    received_data = np.mod(encoded_data + error, 2)

    start_time = time.time()
    output = decode(received_data, p, information_locations, LUTs)
    print(time.time() - start_time)
    print(np.linalg.norm(output - embedded_data))
