from predict import EXPERIMENT_PREDICTION_OUTPUT_DIR
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import os


proj_dir = EXPERIMENT_PREDICTION_OUTPUT_DIR
exp_names = ['snr-10', 'snr-7', 'snr-3', 'snr0', 'snr3', 'snr7', 'snr10']
stat_fname = 'eval_results_.json'
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']


def draw_by_snr_separate():
    fig = plt.figure(figsize=(7, 12))
    ax1 = fig.add_subplot(7, 1, 1)
    ax2 = fig.add_subplot(7, 1, 2)
    ax3 = fig.add_subplot(7, 1, 3)
    ax4 = fig.add_subplot(7, 1, 4)
    ax5 = fig.add_subplot(7, 1, 5)
    ax6 = fig.add_subplot(7, 1, 6)
    ax7 = fig.add_subplot(7, 1, 7)
    plt.tight_layout()

    snrs = []
    avg_l1 = []
    avg_stoi = []
    avg_csig = []
    avg_cbak = []
    avg_covl = []
    avg_pseq = []
    avg_ssnr = []
    for j, name in enumerate(exp_names):
        path = os.path.join(proj_dir, stat_fname.split('.json')[0] + name + '.json')
        print(path)

        num_videos = 0
        snr = 0
        stats = []
        files = []
        with open(path, 'r') as fp:
            info = json.load(fp)
            num_videos = info['num_videos']
            snr = info['snr']
            stats = info['denoise_statistics']
            files = info['files']

        snrs.append(snr)
        avg_l1.append(stats['avg_l1'])
        avg_stoi.append(stats['avg_stoi'])
        avg_csig.append(stats['avg_csig'])
        avg_cbak.append(stats['avg_cbak'])
        avg_covl.append(stats['avg_covl'])
        avg_pseq.append(stats['avg_pesq'])
        avg_ssnr.append(stats['avg_ssnr'])

    ax1.plot(snrs, avg_l1, marker='o')
    ax1.set_title('Average L1')
    ax2.plot(snrs, avg_stoi, marker='o')
    ax2.set_title('Average STOI')
    ax3.plot(snrs, avg_csig, marker='o')
    ax3.set_title('Average CSIG')
    ax4.plot(snrs, avg_cbak, marker='o')
    ax4.set_title('Average CBAK')
    ax5.plot(snrs, avg_covl, marker='o')
    ax5.set_title('Average COVL')
    ax6.plot(snrs, avg_pseq, marker='o')
    ax6.set_title('Average PESQ')
    ax7.plot(snrs, avg_ssnr, marker='o')
    ax7.set_title('Average SSNR')


    plt.savefig('stat_plot_all_exp_snr.png')


def draw_by_snr_agg():
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=None, hspace=None)
    plt.title('Average Metrics')
    plt.xlabel('SNR')

    snrs = []
    avg_l1 = []
    avg_stoi = []
    avg_csig = []
    avg_cbak = []
    avg_covl = []
    avg_pseq = []
    avg_ssnr = []
    for j, name in enumerate(exp_names):
        path = os.path.join(proj_dir, stat_fname.split('.json')[0] + name + '.json')
        print(path)

        num_videos = 0
        snr = 0
        stats = []
        files = []
        with open(path, 'r') as fp:
            info = json.load(fp)
            num_videos = info['num_videos']
            snr = info['snr']
            stats = info['denoise_statistics']
            files = info['files']

        snrs.append(snr)
        avg_l1.append(stats['avg_l1'])
        avg_stoi.append(stats['avg_stoi'])
        avg_csig.append(stats['avg_csig'])
        avg_cbak.append(stats['avg_cbak'])
        avg_covl.append(stats['avg_covl'])
        avg_pseq.append(stats['avg_pesq'])
        avg_ssnr.append(stats['avg_ssnr'])

    plt.plot(snrs, avg_l1, marker='o', label='Average L1')
    plt.plot(snrs, avg_stoi, marker='o', label='Average STOI')
    plt.plot(snrs, avg_csig, marker='o', label='Average CSIG')
    plt.plot(snrs, avg_cbak, marker='o', label='Average CBAK')
    plt.plot(snrs, avg_covl, marker='o', label='Average COVL')
    plt.plot(snrs, avg_pseq, marker='o', label='Average PESQ')
    plt.plot(snrs, avg_ssnr, marker='o', label='Average SSNR')

    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('stat_plot_all_exp_snr_agg.png')


def draw_by_snr_rundi():
    fig = plt.figure(figsize=(5, 12))
    ax1 = fig.add_subplot(4, 1, 1)
    plt.title('avg l1')

    ax2 = fig.add_subplot(4, 1, 2)
    plt.title('avg pesq')

    ax3 = fig.add_subplot(4, 1, 3)
    plt.title('avg ssnr')

    ax4 = fig.add_subplot(4, 1, 4)
    plt.title('number of samples')


    for j, name in enumerate(exp_names):
        path = os.path.join(proj_dir, stat_fname.split('.json')[0] + name + '.json')
        print(path)

        with open(path, 'r') as fp:
            info = json.load(fp)['files']

        ratios = [item['bitstream'].count('0') / len(item['bitstream']) for item in info]
        snrs = [int(item['snr']) for item in info]
        cost_l1 = [item['l1'] for item in info]
        cost_pseq = [item['pesq'] for item in info]
        cost_ssnr = [item['ssnr'] for item in info]
        # print('ratios:', ratios)
        # print('snrs:', snrs)
        # print('cost_l1:', cost_l1)
        # print('cost_pseq:', cost_pseq)
        # print('cost_ssnr:', cost_ssnr)

        bins_l1 = [0] * 7
        bins_pseq = [0] * 7
        bins_ssnr = [0] * 7
        nums = [0] * 7
        snr2bin = {"-10":0, "-7":1, "-3":2, "0":3, "3":4, "7":5, "10":6}

        for i in range(len(cost_l1)):
            idx = snr2bin[str(snrs[i])]
            bins_l1[idx] += cost_l1[i]
            bins_pseq[idx] += cost_pseq[i]
            bins_ssnr[idx] += cost_ssnr[i]
            nums[idx] += 1

        nums = np.asarray(nums)
        bins_l1 = np.asarray([bins_l1[i] / (nums[i] + 1e-10) for i in range(len(nums))])
        bins_pseq = np.asarray([bins_pseq[i] / (nums[i] + 1e-10) for i in range(len(nums))])
        bins_ssnr = np.asarray([bins_ssnr[i] / (nums[i] + 1e-10) for i in range(len(nums))])

        ax1.plot([-10, -7, -3, 0, 3, 7, 10], bins_l1, marker='o', c=colors[j], label=exp_names[j])
        ax1.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand")
        ax2.plot([-10, -7, -3, 0, 3, 7, 10], bins_pseq, marker='o', c=colors[j])
        ax3.plot([-10, -7, -3, 0, 3, 7, 10], bins_ssnr, marker='o', c=colors[j])
        ax4.plot([-10, -7, -3, 0, 3, 7, 10], nums, marker='o', c=colors[j])

    plt.savefig('stat_plot_all_exp_snr.png')


def draw_by_ratio_rundi():
    fig = plt.figure(figsize=(5, 12))
    ax1 = fig.add_subplot(4, 1, 1)
    plt.title('avg l1')

    ax2 = fig.add_subplot(4, 1, 2)
    plt.title('avg pesq')

    ax3 = fig.add_subplot(4, 1, 3)
    plt.title('avg ssnr')

    ax4 = fig.add_subplot(4, 1, 4)
    plt.title('number of samples')


    for j, name in enumerate(exp_names):
        path = os.path.join(proj_dir, stat_fname.split('.json')[0] + name + '.json')
        print(path)

        with open(path, 'r') as fp:
            info = json.load(fp)['files']

        ratios = [item['bitstream'].count('0') / len(item['bitstream']) for item in info]
        snrs = [item['snr'] for item in info]
        cost_l1 = [item['l1'] for item in info]
        cost_pseq = [item['pesq'] for item in info]
        cost_ssnr = [item['ssnr'] for item in info]
        # print('ratios:', ratios)
        # print('snrs:', snrs)
        # print('cost_l1:', cost_l1)
        # print('cost_pseq:', cost_pseq)
        # print('cost_ssnr:', cost_ssnr)

        chunk_size = 0.04

        bins_l1 = [0] * int(1 / chunk_size)
        bins_pseq = [0] * int(1 / chunk_size)
        bins_ssnr = [0] * int(1 / chunk_size)
        nums = [0] * int(1 / chunk_size)

        print(min(ratios), max(ratios))

        for i in range(len(ratios)):
            iid = int(ratios[i] / chunk_size)
            bins_l1[iid] += cost_l1[i]
            bins_pseq[iid] += cost_pseq[i]
            bins_ssnr[iid] += cost_ssnr[i]
            nums[iid] += 1
        print('bins_l1:', bins_l1)
        print('bins_pseq:', bins_pseq)
        print('bins_ssnr:', bins_ssnr)
        print('nums:', nums)

        nums = np.asarray(nums)
        bins_l1 = np.asarray([bins_l1[i] / (nums[i] + 1e-10) for i in range(len(nums))])
        bins_pseq = np.asarray([bins_pseq[i] / (nums[i] + 1e-10) for i in range(len(nums))])
        bins_ssnr = np.asarray([bins_ssnr[i] / (nums[i] + 1e-10) for i in range(len(nums))])
        ratios = np.asarray([x * chunk_size for x in list(range(len(nums)))])
        # print('bins_l1:', bins_l1)
        # print('bins_pseq:', bins_pseq)
        # print('bins_ssnr:', bins_ssnr)
        # print('ratios:', ratios)

        valid_idx = np.where(nums >= 1)[0]

        nums = nums[valid_idx]
        bins_l1 = bins_l1[valid_idx]
        bins_pseq = bins_pseq[valid_idx]
        bins_ssnr = bins_ssnr[valid_idx]
        ratios = ratios[valid_idx]

        ax1.plot(ratios, bins_l1, marker='o', c=colors[j], label=exp_names[j])
        ax1.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand")
        ax2.plot(ratios, bins_pseq, marker='o', c=colors[j])
        ax3.plot(ratios, bins_ssnr, marker='o', c=colors[j])
        ax4.plot(ratios, nums, marker='o', c=colors[j])

    plt.savefig('stat_plot_all_exp_ratio.png')


if __name__ == '__main__':
    # draw_by_snr_agg()
    draw_by_snr_separate()
    # draw_by_ratio()
