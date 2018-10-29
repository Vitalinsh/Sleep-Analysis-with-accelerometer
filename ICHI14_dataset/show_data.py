#!/usr/bin/env python
# show_data.py
# author: Marko Borazio for ESS, TU Darmstadt
# January 2014
# description:
# Plot of PSG, raw data and light measurements

from datetime import datetime as dtime
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import csv
from array import array
import glob
import sys
import pdb

datapath = 'data/'


########################################################################
def plot_sleep_phases(pdta, p_no):
    # for displaying reasons, assign awake = 6 to 8,
    # movement = 7 to 6
    pdta.gt[np.where(pdta.gt == 6)] = 8
    pdta.gt[np.where(pdta.gt == 7)] = 6
    tt = pdta.t

    plt.rcParams.update({'font.size': 8})
    plt.rc('figure', figsize=(10, 3), dpi=85, facecolor='w', edgecolor='k')

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.5, 1])

    # ground truth
    ax1 = plt.subplot(gs[0])
    ax1.plot_date(tt, pdta.gt, '-')
    ax1.set_ylim([0, 9])
    ax1.set_yticklabels(['', '3', '2', '1', '', 'R', 'M', ' ', 'W', ''])
    ax1.set_xticklabels('')
    ax1.xaxis.grid(True)
    plt.tick_params(axis='y', which='major', labelsize=6)

    # raw sensor readings
    ax2 = plt.subplot(gs[1])
    ax2.plot_date(pdta.t, pdta.x, '-')
    ax2.plot_date(pdta.t, pdta.y, '-')
    ax2.plot_date(pdta.t, pdta.z, '-')
    ax2.set_yticklabels('')
    ax2.set_xticklabels('')
    ax2.set_ylabel('acc')
    ax2.xaxis.grid(True)

    # light sensor values
    ax3 = plt.subplot(gs[2])
    ax3.plot_date(pdta.t, pdta.l, '-', color='yellow')
    ax3.fill_between(pdta.t, pdta.l, 0, color='yellow')
    ax3.set_yticklabels('')
    ax3.set_ylabel('light')
    ax3.xaxis.grid(True)
    ax3.set_facecolor("gray")

    xfmt = mdates.DateFormatter('%H:%M')
    ax3.xaxis.set_major_formatter(xfmt)

    plt.show()
    plt.close()


########################################################################
# returns a list of all patient IDs
def get_userlist():
    return ['002', '003', '005', '007', '08a', '08b', '09a', '09b', '10a', '011', '013', '014', '15a', '15b', '016',
            '017', '018', '019', '020', '021', '022', '023', '025', '026', '027', '028', '029', '030', '031', '032',
            '033', '034', '035', '036', '037', '038', '040', '042', '043', '044', '045', '047', '048', '049', '051']


########################################################################
###--MAIN--###
def main():
    print(sys.argv, len(sys.argv))

    patient = sys.argv[1]

    print(patient)
    dta = np.load("%sp%s.npy" % (datapath, patient)).view(np.recarray)
    plot_sleep_phases(dta, patient)


########################################################################
if __name__ == "__main__":
    main()
