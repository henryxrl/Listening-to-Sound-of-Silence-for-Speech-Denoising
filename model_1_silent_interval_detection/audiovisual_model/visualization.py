import io
import os
import librosa
from librosa import display
import matplotlib
matplotlib.use('agg', warn=False, force=True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import cv2
import numpy as np
import subprocess


def draw_waveform(signals, sr=16000, titles=None):
    """draw waveform of signal"""
    if not isinstance(signals, list):
        signals = [signals]

    if titles is not None:
        if not isinstance(titles, list):
            titles = [titles]

        if len(signals) != len(titles):
            titles = None

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
    n = len(signals)
    for i, sig in enumerate(signals):
        ax = fig.add_subplot(n, 1, i + 1)
        display.waveplot(sig, sr=sr, ax=ax)
        if titles is not None:
            ax.set_title(titles[i], x=-0.07, y=0.4, va='center', ha='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    arr = buf2ndarray(buf)
    return arr


def draw_waveform_animated_better_quality(animation_path, signals, sr=16000, titles=None, fps=30):
    """draw waveform of signal"""
    if not isinstance(signals, list):
        signals = [signals]

    if titles is not None:
        if not isinstance(titles, list):
            titles = [titles]

        if len(signals) != len(titles):
            titles = None

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
    n = len(signals)
    lines = []
    for i, sig in enumerate(signals):
        ax = fig.add_subplot(n, 1, i + 1)
        display.waveplot(sig, sr=sr, ax=ax)
        if titles is not None:
            ax.set_title(titles[i], x=-0.07, y=0.4, va='center', ha='right')

        # draw animated line
        X_VALS = np.linspace(0, len(sig)/sr, num=int(len(sig)/sr*fps))
        padding = 0
        min_line = min(sig) - padding
        max_line = max(sig) + padding
        l, v = plt.plot(X_VALS[0], min_line, X_VALS[-1], max_line, linewidth=2, color='#ff0000')
        lines.append(l)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(num):
        i = X_VALS[num]
        for line in lines:
            line.set_data([i, i], [-1, 1])
        return lines

    line_anim = animation.FuncAnimation(fig, animate, frames=len(X_VALS), init_func=init, blit=True)

    # save animated plot
    # writer = FFMpegWriter(fps=fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    print('Saving animated plot to', animation_path)
    line_anim.save(animation_path, writer=writer)
    print('Done')


def draw_waveform_animated_faster(animation_path, signals, sr=16000, titles=None, fps=30):
    """draw waveform of signal"""
    if not isinstance(signals, list):
        signals = [signals]

    if titles is not None:
        if not isinstance(titles, list):
            titles = [titles]

        if len(signals) != len(titles):
            titles = None

    fig = plt.figure(figsize=(18, 9))
    canvas_width, canvas_height = fig.canvas.get_width_height()
    fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
    n = len(signals)
    lines = []
    for i, sig in enumerate(signals):
        ax = fig.add_subplot(n, 1, i + 1)
        display.waveplot(sig, sr=sr, ax=ax)
        if titles is not None:
            ax.set_title(titles[i], x=-0.07, y=0.4, va='center', ha='right')

        # draw animated line
        X_VALS = np.linspace(0, len(sig)/sr, num=int(len(sig)/sr*fps))
        padding = 0
        min_line = min(sig) - padding
        max_line = max(sig) + padding
        l, v = plt.plot(X_VALS[0], min_line, X_VALS[-1], max_line, linewidth=2, color='#ff0000')
        lines.append(l)

    def animate(num):
        i = X_VALS[num]
        for line in lines:
            line.set_data([i, i], [-1, 1])

    # Open an ffmpeg process
    cmdstring = ('ffmpeg', 
                 '-y', '-r', '%d' % fps, # overwrite, 1fps
                 '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                 '-pix_fmt', 'argb', # format
                 '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                #  '-q:v', '5', # video quality when -vcodec=mpeg4
                 '-vcodec', 'h264', animation_path) # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    # Draw frames and write to the pipe
    for frame in range(len(X_VALS)):
        # draw the frame
        animate(frame)
        fig.canvas.draw()

        # extract the image as an ARGB string
        string = fig.canvas.tostring_argb()

        # write to pipe
        p.stdin.write(string)

    # Finish up
    p.communicate()


def draw_spectrum(signals, sr=16000, n_fft=510, hop_length=158, win_length=400, titles=None):
    """draw waveform of signal"""
    if not isinstance(signals, list):
        signals = [signals]

    if titles is not None:
        if not isinstance(titles, list):
            titles = [titles]

        if len(signals) != len(titles):
            titles = None

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
    n = len(signals)
    for i, sig in enumerate(signals):
        ax = fig.add_subplot(n, 1, i + 1)
        s_spec = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        s_mag = np.abs(s_spec)
        display.specshow(librosa.amplitude_to_db(s_mag, ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax)
        if titles is not None:
            ax.set_title(titles[i], x=-0.07, y=0.4, va='center', ha='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    arr = buf2ndarray(buf)
    return arr


def buf2ndarray(buffer):
    """buffer to img ndarray. (H, W, 3) BGR"""
    return np.asarray(cv2.imdecode(np.fromstring(buffer.read(), np.uint8), 1))


def arr2colormap(arr, path=None):
    arr_normalized = (np.tanh(arr) + 1) / 2 * 255
    img = arr_normalized.astype(np.uint8)  # cv2.applyColorMap(arr_normalized.astype(np.uint8), 2)
    if path is not None:
        cv2.imwrite(path, img)
    return img


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)) * 255
    grayscale_im = cv2.applyColorMap(grayscale_im.astype(np.uint8), 2)
    # grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im.astype(np.uint8)


if __name__ == '__main__':
    src_dir = "/home/bourgan/wurundi/dataset/DS_segments/clean_testset_wav_segs/p232/p232_001/p232_001__000.wav"
    mixed_signal, sr = librosa.load(src_dir, sr=16000)
    from tensorboardX import SummaryWriter
    tb = SummaryWriter('test.events')
    tb.add_scalar("scale", 1.2, global_step=2)
    tb.add_audio("audio", np.asarray(mixed_signal)[np.newaxis, :], global_step=2, sample_rate=16000)
