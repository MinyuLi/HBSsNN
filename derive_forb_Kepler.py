import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm
import logging
import os
import sys
import pandas as pd
import time as tm
import shutil
from multiprocessing import Pool
import signal
import traceback
import math
import lightkurve as lk


olddpi = rcParams["savefig.dpi"]
rcParams["savefig.dpi"] = 300
rcParams['font.size'] = 8
unit = 1000
sn_level=4.

def calc_max_likelihood(freqs_intervals):
	mean = np.mean(freqs_intervals)
	std_dev = np.std(freqs_intervals)
	print("std_dev={}, mean={}".format(std_dev,mean))

	filtered_data = [x for x in freqs_intervals if (mean - std_dev) <= x <= (mean + std_dev)]
	filtered_data = freqs_intervals
	filtered_mean = np.mean(filtered_data)
	filtered_std_dev = np.std(filtered_data)

	x = np.linspace(min(filtered_data), max(filtered_data), 100)
	y = norm.pdf(x, filtered_mean, filtered_std_dev)

	plt.hist(filtered_data, bins=20, density=True, alpha=0.6, color='g')
	plt.plot(x, y, 'r-', linewidth=2)
	plt.title('Gaussian Distribution of Filtered Data')
	plt.xlabel('Frequency')
	plt.ylabel('Density')

	plt.show()

	print(f"mu: {filtered_mean}, sigma: {filtered_std_dev}")
	return filtered_mean

def calc_nyquist_and_freq_resolution(lc_path):
	t = np.loadtxt(lc_path, usecols=(0), dtype="float", unpack=True, comments='#')
	space = t[1:]-t[0:-1]
	deltaT = (t[-1]-t[0])/(len(t)-1)
	return 1./(2*np.median(space)) , 1./(t[-1]-t[0])

def toSpectra(lc_path, maxf):
	if maxf > 24:
		maxf = 24
	strMaxf = str(maxf)
	trf_file = lc_path[:-3] + 'trf'
	max_file = lc_path[:-3] + 'max'
	fnpeaks = os.path.join('.', 'fnpeaks', 'fnpeaks_phase')
	command = fnpeaks + ' -f ' + '\'' + lc_path + '\'' + ' 0 '+strMaxf+' 0.00010'
	logging.info(f"begin:{command}")
	os.system(command)
	return trf_file, max_file

def get_harmonic(f_p, f, error):
	times = f/f_p
	round_times = round(times)
	if np.abs(times - round_times) < error:
		return round_times
	return -1

def filter_side_lobe(freq, amp, freq_resolution):
	valid_freqs=[0.0]
	valid_amps=[0.0]
	for i in range(len(freq)):
		
		f1 = freq[i]
		amp1=amp[i]
		is_side_lobe = False
		for j in range(len(valid_freqs)):
			f2 = valid_freqs[j]
			if np.abs(f1-f2) < freq_resolution:
				is_side_lobe = True
				break
		if is_side_lobe == False:
			valid_freqs.append(f1)
			valid_amps.append(amp1)
	return valid_freqs, valid_amps


def try_forb(freq_lst, amp_lst, f0, freq_resolution):
	forb_cadicate=[]
	harmonic_num=[]
	match_num=[]
	amp_num=[]
	logging.info("f0 num={}\n {}".format(len(f0), f0))
	should_check_match_cnt = False
	for i in range(26):
		for k in range(len(f0)):
			cadidate_forb = f0[k]/(i+1)
			#if cadidate_forb < 0.037:
			#	break
			valid_freqs, valid_amps = filter_side_lobe(freq_lst, amp_lst, cadidate_forb*0.45)
			#logging.info("valid freqs before={} after={}".format(len(freq_lst), len(valid_freqs)))
			harmonic_lst = []
			amp_sum = 0.0
			for j in range(len(valid_freqs)):
				harmonic = get_harmonic(cadidate_forb, valid_freqs[j], 0.1)
				if harmonic > 0 and harmonic not in harmonic_lst and harmonic <= 100:
					harmonic_lst.append(harmonic)
					amp_sum += valid_amps[j]
			match_cnt = len(harmonic_lst)
			max_harmonic = min(100, round(np.max(freq_lst)/cadidate_forb))
			miss_cnt = max_harmonic - match_cnt
			#if miss_cnt == 0 and i != 0:
			#	logging.info("Find forb={}.".format(cadidate_forb))
			#	return cadidate_forb
			#logging.info("max freq={} expect i={} total harmonic={}".format(np.max(freq_lst), i, round(np.max(freq_lst)/cadidate_forb)))
			match_ratio = match_cnt / max_harmonic
			if match_ratio >= 1 or i == 0:
				match_ratio = match_cnt / (max_harmonic+1)

			harmonic_num.append(match_ratio)
			match_num.append(match_cnt)
			forb_cadicate.append(cadidate_forb)
			amp_num.append(amp_sum)
	logging.info(harmonic_num)
	logging.info(amp_num)
	logging.info(match_num)
	
	harmonic_num = np.array(harmonic_num)
	match_num = np.array(match_num)
	forb_cadicate = np.array(forb_cadicate)
	amp_num = np.array(amp_num)
	#logging.info("max amp sum:{}".format(np.max(amp_num)))
	pos = np.where(harmonic_num >= np.max(harmonic_num)*0.9)[0]
	if len(pos) == 1:
		return forb_cadicate[pos[0]], harmonic_num[pos[0]]

	harmonic_num = harmonic_num[pos]
	match_num = match_num[pos]
	forb_cadicate = forb_cadicate[pos]
	pos = np.where(match_num == np.max(match_num))[0][0]
	return forb_cadicate[pos], harmonic_num[pos]

def round_to_nearest_step(value, step=0.0001):
	"""Round a value to the nearest multiple of the given step."""
	return round(value / step) * step

def gcd_of_list(numbers):
	"""Find the greatest common divisor (GCD) of a list of numbers."""
	if not numbers:
		return None
	current_gcd = numbers[0]
	for number in numbers[1:]:
		current_gcd = math.gcd(current_gcd, int(number))
	return current_gcd

def calculate_gcd_with_error(freq_lst, error_margin=0.0001):
	"""
	Calculate the greatest common divisor (GCD) of a list of frequencies,
	considering an error margin.
	
	Parameters:
	freq_lst (list of float): List of frequency values.
	error_margin (float): The error margin within which two frequencies are considered equal.
	
	Returns:
	float: The GCD of the rounded frequencies.
	"""
	# Round each frequency to the nearest multiple of the error margin
	rounded_freqs = [round_to_nearest_step(freq, error_margin) for freq in freq_lst]
	
	# Convert rounded frequencies to integers by scaling up
	scale_factor = 1 / error_margin
	integer_freqs = [int(rounded_freq * scale_factor) for rounded_freq in rounded_freqs]
	
	# Remove duplicates
	unique_integer_freqs = list(set(integer_freqs))
	
	# Calculate the GCD of the unique integer frequencies
	gcd_value = gcd_of_list(unique_integer_freqs)
	
	# Scale back to the original frequency range
	gcd_frequency = gcd_value / scale_factor
	
	return gcd_frequency

def find_most_frequent_frequency(frequency_intervals, error_margin=0.0001):
	"""
	Find the most frequent frequency in the given array, considering an error margin.
	
	Parameters:
	frequency_intervals (np.array): Array of frequency values.
	error_margin (float): The error margin within which two frequencies are considered equal.
	
	Returns:
	float: The most frequent frequency.
	"""
	# Round each frequency to the nearest multiple of the error margin
	rounded_freqs = [round_to_nearest_step(freq, error_margin) for freq in frequency_intervals]
	
	# Count the occurrences of each rounded frequency
	freq_counter = Counter(rounded_freqs)
	
	# Find the most common frequency
	most_common_freq, _ = freq_counter.most_common(1)[0]
	
	return most_common_freq



def calc_forb(freq_lst, amp_lst, f0,freq_resolution):
	valid_freqs, valid_amps = filter_side_lobe(freq_lst, amp_lst, freq_resolution)
	logging.info("freq_lst len={}, valid_freqs len={}, freq_resolution={}".format(len(freq_lst), len(valid_freqs),freq_resolution))
	#logging.info(freq_lst)
	#valid_cnt = min(len(valid_freqs), 10)
	#valid_freqs = valid_freqs[0:valid_cnt]
	freqs = np.array(sorted(valid_freqs))
	#freqs = freqs[0:10]
	#print(freqs)
	'''
	gcd_frequency = calculate_gcd_with_error(freqs,freq_resolution)

	#In the case of self-excited gamma Doradus, the gcd_frequency is not the correct forb.
	total_harmonics = int(np.max(freqs) / gcd_frequency)
	matched_harmonics = len(freqs)
	missed_harmonics = total_harmonics - matched_harmonics
	if missed_harmonics < matched_harmonics:
		return gcd_frequency
	'''
	freqs_intervals = freqs[1:] - freqs[0:-1]
	logging.info(freqs_intervals)
	judge_reso = 0.005
	freqs_intervals = freqs_intervals[freqs_intervals>10*freq_resolution]
	if len(freqs_intervals) == 0:
		return -1
	most_frequent_frequency = find_most_frequent_frequency(freqs_intervals,freq_resolution)
	logging.info("most_frequent_frequency={}".format(most_frequent_frequency))
	new_freq=[]
	for i in range(len(freqs_intervals)):
		if np.abs(freqs_intervals[i] - most_frequent_frequency) < freq_resolution:
			new_freq.append(freqs_intervals[i])
	new_freq = np.array(new_freq)
	mean_forb = np.mean(new_freq)
	#most_frequent_frequency = calc_max_likelihood(freqs_intervals)
	return np.mean(new_freq)

def get_freq_cluster(freq1):
	prominent = freq1[0]
	L1 = prominent*(1-0.02) < freq1
	L2 = freq1 < prominent*1.02
	freq_cluster = freq1[L1&L2]
	if len(freq_cluster) > 1:
		freq_cluster = freq_cluster[0:1]
	return freq_cluster

def get_niose(freq, freq_all, amp_all):
	box = 1
	L = (freq_all <= freq+box) & (freq_all >= freq-box)
	freqs = freq_all[L]
	amps = amp_all[L]
	return np.mean(amps)

def extractHarmonics(max_file, trf_file, freq_resolution, judge):
	freq_all, amp_all, phase_all = np.loadtxt(trf_file, usecols=(0,1,2), dtype="float", unpack=True, ndmin=2)
	no, freq_max, period, amp_max, phase, sn_max = np.loadtxt(max_file, usecols=(0,1,2,3,4,5), dtype="float", unpack=True, comments=r'%')

	noise_calc = np.array(sn_max)*0
	sn_calc = np.array(sn_max)*0
	for i in range(len(no)):
		noise = get_niose(freq_max[i], freq_all, amp_all)
		noise_calc[i] = noise
		sn_calc[i] = amp_max[i]/noise

	L = sn_calc >= sn_level
	L2 = freq_max > 0.004
	no1 = no[L & L2]
	freq1 = freq_max[L & L2]
	amp1 = amp_max[L & L2]
	freq_cluster = get_freq_cluster(freq1)
	forb, match_ratio = try_forb(freq1, amp1, freq_cluster,freq_resolution)

	harmonic_lst=[]
	for i in range(len(no1)):
		harm = get_harmonic(forb, freq1[i], 0.03)
		if harm > 0 and harm not in harmonic_lst and harm <= 100:
			harmonic_lst.append(harm)
	match_cnt = len(harmonic_lst)
	max_harmonic = min(100, round(np.max(freq1)/forb))
	logging.info("forb1={} match_cnt1={} match_ratio={}".format(forb, match_cnt, match_ratio))
	L = freq_max.argsort()
	max_freq = freq_max[L]
	max_amp = amp_max[L]
	max_sn = sn_calc[L]
	if judge == True and match_ratio >= 0.8:
		return harmonic_lst, max_freq, max_amp/max_sn, forb, match_cnt, match_ratio
	
	forb2 = calc_forb(freq1, amp1, freq_cluster,freq_resolution)
	if forb2 < 0:
		return harmonic_lst, max_freq, max_amp/max_sn, forb, match_cnt, match_ratio

	harmonic_lst2=[]
	for i in range(len(no1)):
		harm = get_harmonic(forb2, freq1[i], 0.03)
		if harm > 0 and harm not in harmonic_lst2 and harm <= 100:
			harmonic_lst2.append(harm)
	match_cnt2 = len(harmonic_lst2)
	max_harmonic = min(100, round(np.max(freq1)/forb2))
	match_ratio2 = match_cnt2 / max_harmonic
	logging.info("forb2={} match_cnt2={} match_ratio={}".format(forb2, match_cnt2, match_ratio2))

	if match_cnt >= match_cnt2:
		return harmonic_lst, max_freq, max_amp/max_sn, forb, match_cnt, match_ratio
	else:
		return harmonic_lst2, max_freq, max_amp/max_sn, forb2, match_cnt2, match_ratio2


def plot_spectrum(trf_file, harmonics, obj_id, f, file_path, max_freq, max_noise):
	freq, amp, phase = np.loadtxt(trf_file, usecols=(0,1,2), dtype="float", unpack=True)
	fig, axs = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
	axes1 = axs
	x_min = 0.
	x_max = np.max(freq)
	y_min = 0.
	y_max = np.max(amp)*unit*1.2
	harmonic_num = int(x_max/f)+1
	for i in range(harmonic_num):
		l=i+1
		if l in harmonics:
			line_type = 'b--'
		else:
			line_type = 'r--'
		axes1.plot([l*f, l*f], [y_min, y_max], line_type, lw=0.4)
	axes1.plot(freq, amp*unit, 'k', ms=1, lw=0.5)#, alpha=0.8)
	axes1.plot(max_freq, max_noise*unit*sn_level, 'r', ms=1, lw=0.5)
	title = obj_id + r", $\mathit{f}$"+ r"$_{\rm orb}$"+ "={0:.8f}".format(f)
	axes1.set_title(title)
	axes1.set_ylabel('Amplitude (mmag)')
	axes1.set_xlabel('Frequency  (d$^{-1}$)')
	
	#axes1.set_xscale("log", nonpositive='clip')
	#axes1.set_yscale("log", nonpositive='clip')

	axes1.set_xlim(x_min, x_max)
	axes1.set_ylim(y_min, y_max)

	#plt.tight_layout()
	plt.savefig(file_path)
	plt.close(fig)

def derive_forb(obj_id, lc_path, result_path):
	nyquist, freq_resolution = calc_nyquist_and_freq_resolution(lc_path)
	logging.info("{} max_freq={} nyquist={} freq_resolution={}".format(obj_id, nyquist, nyquist, freq_resolution) )
	trf_file, max_file = toSpectra(lc_path, 10)
	harmonics, max_freq, max_noise,forb, match_cnt, match_ratio = extractHarmonics(max_file, trf_file, freq_resolution, True)

	if match_ratio < 0.6:
		trf_file, max_file = toSpectra(lc_path, forb*110)
		harmonics2, max_freq2, max_noise2,forb2, match_cnt2, match_ratio2 = extractHarmonics(max_file, trf_file, freq_resolution, False)
		if match_ratio2 > match_ratio:
			harmonics, max_freq, max_noise,forb, match_cnt, match_ratio = harmonics2, max_freq2, max_noise2,forb2, match_cnt2, match_ratio2

	rst = "{0:s} ".format(obj_id) +\
		"{0:.8f} ".format(forb)
		

	rst = rst + "\n"
	rstfile =  os.path.join(result_path, "samples.dat") #The results are collected in one file
	with open(rstfile, 'a') as fw:
		fw.write(rst)

	file_path = os.path.join(result_path, obj_id+'.png')
	plot_spectrum(trf_file, harmonics, obj_id, forb, file_path,max_freq, max_noise)

	os.remove(trf_file)
	os.remove(max_file)

def asyncExtractFeatures(param):
	try:
		lc_dir, lc_filename, result_path = param
		pars = lc_filename.split('.')
		obj_id = pars[0]
		obj_id = obj_id.lstrip('0')

		lc_path = os.path.join(lc_dir, lc_filename)
		derive_forb(obj_id, lc_path, result_path)
	except Exception as e:
		exc_type, exc_val, exc_tb = sys.exc_info()
		error_msg = "".join( traceback.format_exception( exc_type, exc_val, exc_tb) )
		logging.error(f"{lc_filename} failed. except:{str(e)}.\n{error_msg}")
		return
		
	#If there is a memory leak, uncomment this.
	#os._exit(0)


def deriveLCs(result_path, lc_dir):
	cpu_cnt = os.cpu_count()
	po = Pool(1)
	for dirpath, dirnames, filenames in os.walk(lc_dir):
		for filename in filenames:
			if filename[-4:] == ".dat":
				param = (dirpath, filename, result_path)
				po.apply_async(asyncExtractFeatures, (param,))
				#asyncExtractFeatures(param)
		
	po.close()
	po.join()

if __name__ == "__main__":
	fnpeaks_path = os.path.join('.', 'fnpeaks', 'fnpeaks_phase')
	if not os.path.exists(fnpeaks_path):
		raise ValueError('You need to enter the fnpeaks directory and run the make command to build fnpeaks_phase.')
	
	result_path = os.path.join('Results', tm.strftime("%Y-%m-%d %H-%M-%S", tm.localtime())) 
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter(
		fmt='%(asctime)s : %(levelname)s : %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)

	fh = logging.FileHandler(os.path.join(result_path, 'logging.log'), encoding="utf-8", mode="a", delay=False)
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	lc_dir = os.path.join('.', 'lc-Kepler')
	deriveLCs(result_path, lc_dir)


