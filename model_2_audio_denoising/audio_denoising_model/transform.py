import numpy as np
from numpy import inf
import librosa
import torch

N_FFT = 510
HOP_LENGTH = 158
WIN_LENGTH = 400

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D


def real_imag_shrink(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink


def generate_cRM(Y,S):
    '''
    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    '''
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:,:,0],S[:,:,0])+np.multiply(Y[:,:,1],S[:,:,1])
    square_real = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_real = np.divide(M_real,square_real+epsilon)
    M[:,:,0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:,:,0],S[:,:,1])-np.multiply(Y[:,:,1],S[:,:,0])
    square_img = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_img = np.divide(M_img,square_img+epsilon)
    M[:,:,1] = M_img
    return M


def cRM_tanh_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = 1-np.exp(-C*M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1+np.exp(-C*M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator,denominator)

    return crm


def cRM_tanh_recover(O,K=10,C=0.1):
    '''
    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O + 1e-8
    denominator = K+O + 1e-8
    M = -np.multiply((1.0/C),np.log(np.divide(numerator,denominator)))

    return M


def cRM_sigmoid_compress(M, a=0.1, b=0):
    """sigmoid compression"""
    return 1. / (1. + np.exp(-a * M + b))


def cRM_sigmoid_recover(O, a=0.1, b=0):
    """inverse sigmoid"""
    return 1. / a * (np.log(O / (1 - O + 1e-8) + 1e-10) + b)


def fast_cRM(Fclean,Fmix,K=10,C=0.1):
    '''
    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_tanh_compress(M,K,C)
    return crm


def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_tanh_recover(crm,K,C)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S


def fast_cRM_sigmoid(Fclean,Fmix):
    '''
    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_sigmoid_compress(M)
    return crm


def fast_icRM_sigmoid(Y,crm):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_sigmoid_recover(crm)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S


def batch_fast_icRM_sigmoid(Y, crm, a=0.1, b=0):
    """

    :param Y: (B, 2, F, T)
    :param crm: (B, 2, F, T)
    :param a:
    :param b:
    :return:
    """
    M = 1. / a * (torch.log(crm / (1 - crm + 1e-8) + 1e-10) + b)
    r = M[:, 0, :, :] * Y[:, 0, :, :] - M[:, 1, :, :] * Y[:, 1, :, :]
    i = M[:, 0, :, :] * Y[:, 1, :, :] + M[:, 1, :, :] * Y[:, 0, :, :]
    rec = torch.stack([r, i], dim=1)
    return rec


def istft2librosa(S, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    s = real_imag_shrink(S)
    s = librosa.istft(s, hop_length, win_length)
    return s


def power_law(data,power=0.3):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data>=0] = 1
    mask[data<0] = -1
    data = np.power(np.abs(data),power)
    data = data*mask
    return data


def fast_stft(data, power=False, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    if power:
        data = power_law(data)
    return real_imag_expand(librosa.stft(data, n_fft, hop_length, win_length))


def fast_istft(F, power=False, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = librosa.istft(real_imag_shrink(F), hop_length, win_length)
    if power:
        T = power_law(T,(1.0/0.3))
    return T


if __name__ == '__main__':
    clean, _ = librosa.load("/Users/wurundi/PycharmProjects/audiovisual/scripts/test/000000/clean.wav", sr=16000)
    mixed, _ = librosa.load("/Users/wurundi/PycharmProjects/audiovisual/scripts/test/000000/mixed.wav", sr=16000)
    clean = clean # [:48800]
    mixed = mixed # [:48800]
    clean_trans = fast_stft(clean)
    mixed_trans = fast_stft(mixed)
    print(np.max(clean_trans), np.min(clean_trans), np.mean(np.abs(clean_trans)))

    clean_rec = fast_istft(clean_trans)
    mixed_rec = fast_istft(mixed_trans)

    # cRM = generate_cRM(mixed_trans, clean_trans)
    # cRM = fast_cRM(clean_trans, mixed_trans)
    cRM = fast_cRM_sigmoid(clean_trans, mixed_trans)
    print(cRM.shape)
    print(np.max(cRM), np.min(cRM))
    print(cRM)

    # clean_trans_rec = fast_icRM(mixed_trans, cRM)
    clean_trans_rec = fast_icRM_sigmoid(mixed_trans, cRM)
    print(np.mean(np.abs(clean_trans_rec - clean_trans)))
