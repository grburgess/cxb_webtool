#!/usr/bin/env python
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# # Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets

import sys
import os
sys.path.append(os.getcwd())
from scipy import stats
import numpy as np
import gc
from scipy.interpolate import interp1d, RegularGridInterpolator
from astropy.cosmology import WMAP9 as cosmo
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from contextlib import closing
#import gvar
from matplotlib import cm
from astropy.io import ascii
from numba import jit
import scipy.stats as ss
import copy
import pylab as plt
import time
import glob
import math
import gzip

gam_mus=np.arange(1.4,2.21,0.1)
gam_stds=np.arange(0.0,0.51,0.1)
r_mus=np.arange(0.0,2.01,0.2)
r_stds=np.array([0.0,0.1,0.2,0.4,1.0])#np.arange(0.0,0.41,0.1)
ecut_mus=np.array([50,80,110,140,170,200,230,260,290,340,390,440,490])
ecut_stds=np.arange(0,110,20)

#take input from jupyter notebooks
#can be a separate set of parameters for z, Lx, nH
Runabs=np.arange(0,1.8,0.5)#np.asarray([0.,0.3,0.7,1.0,1.3,1.7])
Ecut=np.arange(50.,500.,40)#np.asarray([50,100,150,200,300,500])
Gamma=np.arange(1.4,2.21,0.1)#np.asarray([1.4,1.6,1.8,2.0,2.2])
Gamma_std=np.arange(0.1,0.55,0.1)#np.asarray([0.1,0.2,0.3,0.4,0.5])

# downloaded1 = drive.CreateFile({'id':"1knvjHhTeLy7RikPTOYqWiF59lYjpkEX5"})
# downloaded1.GetContentFile('cdf1.npy.gz') 
foo_c = gzip.GzipFile("cdf1.npy.gz", "r")
#cdfs_c=mf.cdf_func(foo_c)

#downloaded2 = drive.CreateFile({'id':"1rl9VnTVCyotpvWz2Y4XwTlcfM6NFmeYy" }) 
# downloaded2 = drive.CreateFile({'id':"1VFJaG-guQEOX1WAqt9ReRM2TUlLQY-mM" })
# downloaded2.GetContentFile('cdf2.npy.gz') 
foo_r = gzip.GzipFile("cdf2.npy.gz", "r")
#cdfs_r=mf.cdf_func(foo_r)

#downloaded3 = drive.CreateFile({'id':"1ZYJAe7fG2VbQVRxMFLmhnFVm6d_dOkJq" }) 
# downloaded3 = drive.CreateFile({'id':"1MlBUjNcZzBH9jLf_DG2jhKfiXEvAmWTr" }) 
# downloaded3.GetContentFile('cdf3.npy.gz') 
foo_s = gzip.GzipFile("cdf3.npy.gz", "r")
#cdfs_s=mf.cdf_func(foo_s)


def cdf_func_total(unzip_arrays=[foo_c,foo_r,foo_s]):
	arr_list=[]
	for arr in unzip_arrays:
		m=np.load(arr)
		m=stats.chi2.cdf(m, 18)
		arr_list.append(m)
	arr_stack=np.stack(arr_list)
	main_arr=np.amax(arr_stack,0)
	pval_func_c=RegularGridInterpolator((Gamma, Gamma_std, Ecut, Runabs), 
									   arr_list[0], method='linear',
									   bounds_error=False, fill_value=-30000)
	pval_func_r=RegularGridInterpolator((Gamma, Gamma_std, Ecut, Runabs), 
									   arr_list[1], method='linear',
									   bounds_error=False, fill_value=-30000)
	pval_func_s=RegularGridInterpolator((Gamma, Gamma_std, Ecut, Runabs), 
									   arr_list[2], method='linear',
									   bounds_error=False, fill_value=-30000)
	pval_func_aggr=RegularGridInterpolator((Gamma, Gamma_std, Ecut, Runabs), 
									   main_arr, method='linear',
									   bounds_error=False, fill_value=-30000)
	return pval_func_c,pval_func_r,pval_func_s,pval_func_aggr

num_of_fig_lines=5
# 10 threads with 2000 steps takes 600 seconds
glob_eta=0.000005
#Momentum:
glob_beta=0.8
#glob_beta_matr=np.tile(np.array([1,glob_beta**1,glob_beta**2,glob_beta**3,glob_beta**4]),parr_matr).reshape(parr_matr,5)
glob_beta_matr=np.array([1,glob_beta**1,glob_beta**2,glob_beta**3,glob_beta**4])
#step size:
dw=0.0001
#print("LR:", glob_eta, "Momentum:", glob_beta, 'dw:', dw)

global weigh
global steps

en1=np.linspace(0.56,10,7)
en2=np.linspace(12,40,10)
en3=np.linspace(43,100,6)
energyarrold=np.concatenate([en1,en2,en3])

#downloaded2 = drive.CreateFile({'id':"1XaBlO2MXeQDwGvyiQInrewKi0W7IPxvi"})
#downloaded2.GetContentFile('all_matr.npy.gz') 
fui = gzip.GzipFile('all_matr.npy.gz', "r")
all_spectra=np.load(fui)

interpolator_list=[]
for nH_bin in range(0,3):
	for l_bin in range(0,5):
		spec_interoplator=RegularGridInterpolator((ecut_mus,ecut_stds,r_mus,r_stds, 
												   gam_mus, gam_stds, energyarrold), 
										  		   all_spectra[:,:,:,:,:,:,nH_bin,l_bin,:], 
										  		   method='linear', bounds_error=False,
										  		   fill_value=0)
		interpolator_list.append(spec_interoplator)

#galaxy XRB:
#downloaded2 = drive.CreateFile({'id':"18KqCTB6mnTSwloUVvJGDB7HzKATP3n1G"})
#downloaded2.GetContentFile('gal.txt') 
fui = gzip.GzipFile('gal.txt', "r")
galaxy_xrb_table=ascii.read(fui,delimiter=",")
galaxy_xrb=interp1d(galaxy_xrb_table["energy"],galaxy_xrb_table["xrb"],
					bounds_error=False,fill_value=0.)

#galaxy contribution:
galaxy_contri=galaxy_xrb(energyarrold)
#print("total cxb:", summer_cxb_over_energy+galaxy_contri)

#chandra cosmos nico
#downloaded2 = drive.CreateFile({'id':"1KoEnN7S3xii6vWHsMlTPkvmaR6RvwfpD"})
#downloaded2.GetContentFile('chand.csv') 
chandra=ascii.read("chand.csv")
#rxte:
#downloaded2 = drive.CreateFile({'id':"1WrSlU1JXkIRqbz4NEKghhwABKy8qONmT"})
#downloaded2.GetContentFile('rxte.txt') 
rxte=ascii.read('rxte.txt')
#swift
#downloaded2 = drive.CreateFile({'id':"1h2V2QWdnECbEKO5asR6rgL95cv6YhUA2"})
#downloaded2.GetContentFile('swift.txt') 
swift_bat=ascii.read('swift.txt') #energy lower median upper
#Integral XRB:
#downloaded2 = drive.CreateFile({'id':"1ixCO29Mu7GVEkcGJeaJdNFN-CZs50Vqm"})
#downloaded2.GetContentFile('integral.txt') 
integral=ascii.read('integral.txt')
#ASCA:
#downloaded2 = drive.CreateFile({'id':"1S4IdMoXaoUlC4R8NBEtbxo5b0iFi29_Q"})
#downloaded2.GetContentFile('asca.txt') 
asca=ascii.read('asca.txt')
#heao1: gruber 1999
#downloaded2 = drive.CreateFile({'id':"1TjyyuyeK5k_EmBGf0G_sc7gWkPouctCO"})
#downloaded2.GetContentFile('heao1.txt') 
heao1=ascii.read('heao1.txt')
#heao-a2:
#downloaded2 = drive.CreateFile({'id':"1jK0ODM_s47onkDgHXxn_OVSlPLXFjjHm"})
#downloaded2.GetContentFile('heao-a2.txt') 
heao_a2=ascii.read('heao-a2.txt')


#xray data:
#downloaded2 = drive.CreateFile({'id':"13dS5TqRokM1uluXQM54FltSgSKDi0x9d"})
#downloaded2.GetContentFile('xray_data.csv') 
fui = gzip.GzipFile('xray_data.csv', "r")
xray_error_table=ascii.read(fui,delimiter=",")
xray_error_low=interp1d(xray_error_table["x"],xray_error_table["low"],
					bounds_error=False,fill_value=0.)
xray_error_high=interp1d(xray_error_table["x"],xray_error_table["high"],
					bounds_error=False,fill_value=0.)

all_data_x=np.concatenate((chandra['energy'].data, rxte['energy_med'].data, swift_bat['x'].data[:15]))
all_data_y = np.concatenate((chandra['median'].data,rxte['median'].data,swift_bat['y'].data[:15]))
all_data_err = np.concatenate((chandra['upper'].data-chandra['median'].data,
							   rxte['upper'].data-rxte['median'].data,swift_bat['y_err_up'].data[:15]))
data_err_high=xray_error_high(all_data_x)-all_data_y
data_err_low=all_data_y-xray_error_low(all_data_x)

chand_num=len(chandra['energy'].data)
rxte_num=len(rxte['energy_med'].data)
swift_num=15


def pvalue_dataset(data_x, data, data_err_high, data_err_low, model_x, model_y):
	"""
	Calculate cost function and 
	ordinary least squares using
	model and data
	"""
	#what is the format of the input model_info?
	model_shape=model_y.shape
	#if it is updated cxb, it should be (parr_matr, 23)
	xrb_interp=interp1d(model_x, model_y)
	xrb_rel=xrb_interp(data_x)
	diff=xrb_rel-data
	data_err_full=data_err_high
	find_small=np.where(diff<0)
	data_err_full[find_small[0]]=data_err_low[find_small[0]]
	least_sq_terms_mcmc=(diff/(data_err_full))**2
	sum_of_sq_cost_mcmc=np.sum(least_sq_terms_mcmc)
	cdf_val=stats.chi2.cdf(sum_of_sq_cost_mcmc, 18)
	pval=1.-cdf_val
	return pval

def sum_of_squares(data_x, data, data_err, model_x, model_y):
	"""
	Calculate cost function and 
	ordinary least squares using
	model and data
	"""
	#what is the format of the input model_info?
	model_shape=model_y.shape
	#if it is updated cxb, it should be (parr_matr, 23)
	if len(model_shape)==1:
		xrb_interp=interp1d(model_x, model_y)
		xrb_rel=xrb_interp(data_x)
		diff=xrb_rel-data
		data_err_full=data_err_high
		find_small=np.where(diff<0)
		data_err_full[find_small[0]]=data_err_low[find_small[0]]
		#print("xrb_rel", xrb_rel)
		#print("shape of xrb_rel", xrb_rel.shape)
		least_sq_terms_mcmc_full=(diff/data_err_full)**2
		least_sq_terms_mcmc=(diff/(data_err))**2
		lsq_chand=np.sum(least_sq_terms_mcmc[:chand_num])
		lsq_rxte=np.sum(least_sq_terms_mcmc[chand_num:chand_num+rxte_num])
		lsq_swift=np.sum(least_sq_terms_mcmc[chand_num+rxte_num:])
		lsq_chand_full=np.sum(least_sq_terms_mcmc_full[:chand_num])
		lsq_rxte_full=np.sum(least_sq_terms_mcmc_full[chand_num:chand_num+rxte_num])
		lsq_swift_full=np.sum(least_sq_terms_mcmc_full[chand_num+rxte_num:])

		sum_of_sq_cost_mcmc_full=np.sum(least_sq_terms_mcmc_full)
		sum_of_sq_cost_mcmc=np.sum(least_sq_terms_mcmc)
		least_sq_terms_cost=diff**2
		sum_of_sq_cost=np.sum(least_sq_terms_cost)
	elif len(model_shape)==2:
		#print("real step cost")
		xrb_interp=RegularGridInterpolator((np.arange(model_shape[0]), model_x), model_y, 
											method='linear',bounds_error=False, fill_value=0)
		x_start=np.tile(np.arange(model_shape[0]),(len(data_x),1)).T.reshape(model_shape[0]*len(data_x))
		x_start_en=np.tile(data_x,(model_shape[0],1)).reshape(model_shape[0]*len(data_x))
		x_all=[]
		x_all.append(x_start)
		x_all.append(x_start_en)
		x_all=np.array(x_all).T.reshape(model_shape[0],len(data_x),len(model_shape))
		xrb_rel=xrb_interp(x_all)

		diff=xrb_rel-data
		#print("diff",diff)
		find_small=np.where(diff.reshape([-1])<0)
		data_err_full=np.tile(data_err_high,(model_shape[0],1)).reshape([-1])
		data_err_low_holder=np.tile(data_err_low,(model_shape[0],1)).reshape([-1])
		#print("data_err_low",data_err_low_holder.reshape(diff.shape))
		#print("data_err_high",data_err.reshape(diff.shape))
		data_err_full[find_small[0]]=data_err_low_holder[find_small[0]]
		data_err_full=data_err_full.reshape(diff.shape)
		#print("final data err",data_err)
		
		least_sq_terms_mcmc_full=(diff/data_err_full)**2
		least_sq_terms_mcmc=(diff/data_err)**2
		lsq_chand=np.sum(least_sq_terms_mcmc[:,:chand_num],1)
		lsq_rxte=np.sum(least_sq_terms_mcmc[:,chand_num:chand_num+rxte_num],1)
		lsq_swift=np.sum(least_sq_terms_mcmc[:,chand_num+rxte_num:],1)
		lsq_chand_full=np.sum(least_sq_terms_mcmc_full[:,:chand_num],1)
		lsq_rxte_full=np.sum(least_sq_terms_mcmc_full[:,chand_num:chand_num+rxte_num],1)
		lsq_swift_full=np.sum(least_sq_terms_mcmc_full[:,chand_num+rxte_num:],1)

		sum_of_sq_cost_mcmc=np.sum(least_sq_terms_mcmc,1)
		least_sq_terms_cost=diff**2
		sum_of_sq_cost=np.sum(least_sq_terms_cost,1)
	elif len(model_shape)==3:
		#print("derivative")
		xrb_interp=RegularGridInterpolator((np.arange(model_shape[0]), np.arange(model_shape[1]), 
											model_x), model_y, method='linear',
											bounds_error=False, fill_value=0)
		x_start=np.tile(np.arange(model_shape[0]),(model_shape[1]*len(data_x),1)).T.reshape(model_shape[0]*model_shape[1]*len(data_x))
		x_start_2=np.tile(np.arange(model_shape[1]),(len(data_x),1)).T.reshape(model_shape[1]*len(data_x))
		x_start_2=np.tile(x_start_2, (model_shape[0],1)).reshape(model_shape[0]*model_shape[1]*len(data_x))
		
		x_start_en=np.tile(data_x,(model_shape[0]*model_shape[1],1)).reshape(model_shape[0]*model_shape[1]*len(data_x))
		x_all=[]
		x_all.append(x_start)
		x_all.append(x_start_2)
		x_all.append(x_start_en)
		x_all=np.array(x_all).T.reshape(model_shape[0],model_shape[1],len(data_x),len(model_shape))
		#print("x_start",x_all)
		xrb_rel=xrb_interp(x_all)

		diff=xrb_rel-data
		#print("diff",diff)
		find_small=np.where(diff.reshape([-1])<0)
		data_err_full=np.tile(data_err_high,(model_shape[0]*model_shape[1],1)).reshape([-1])
		data_err_low_holder=np.tile(data_err_low,(model_shape[0]*model_shape[1],1)).reshape([-1])
		#print("deriv data_err_low",data_err_low_holder.reshape(diff.shape))
		#print("deriv data_err_high",data_err.reshape(diff.shape))
		data_err_full[find_small[0]]=data_err_low_holder[find_small[0]]
		data_err_full=data_err_full.reshape(diff.shape)
		#print("deriv final data err",data_err)
		
		least_sq_terms_mcmc_full=(diff/data_err_full)**2
		least_sq_terms_mcmc=(diff/(data_err))**2
		lsq_chand=np.sum(least_sq_terms_mcmc[:,:chand_num],2)
		lsq_rxte=np.sum(least_sq_terms_mcmc[:,chand_num:chand_num+rxte_num],2)
		lsq_swift=np.sum(least_sq_terms_mcmc[:,chand_num+rxte_num:],2)
		lsq_chand_full=np.sum(least_sq_terms_mcmc_full[:,:chand_num],2)
		lsq_rxte_full=np.sum(least_sq_terms_mcmc_full[:,chand_num:chand_num+rxte_num],2)
		lsq_swift_full=np.sum(least_sq_terms_mcmc_full[:,chand_num+rxte_num:],2)

		sum_of_sq_cost_mcmc=np.sum(least_sq_terms_mcmc,2)
		least_sq_terms_cost=diff**2
		sum_of_sq_cost=np.sum(least_sq_terms_cost,2)
	
	num_of_data=len(data)
	return num_of_data, sum_of_sq_cost, sum_of_sq_cost_mcmc, \
			[lsq_chand,lsq_rxte,lsq_swift], [lsq_chand_full,lsq_rxte_full,lsq_swift_full]

def xrb_least_squares(model_x, model_y):
	"""
	Packages least squares/cost function for all datasets
	"""
	#Chandra COSMOS: 
	df, total_lsq_cost, total_lsq_mcmc, lsqs, lsqs_full = sum_of_squares(all_data_x, all_data_y, all_data_err, 
						   															model_x, model_y)
	
	return total_lsq_cost,total_lsq_mcmc,df, lsqs, lsqs_full


def update_weights(eta, w):
	"""
	Update the network's weights by applying
	gradient descent using backpropagation.
	"""
	nabla_w = np.zeros(num_params)
	delta_nabla_w = backprop(w)
	steps_now=-eta*delta_nabla_w
	#print("step sizes", steps_now)
	steps_list.append(steps_now)
	steps_with_momentum=momentum()
	#print("After momentum step:", steps_with_momentum)
	weight_holder=w+steps_with_momentum
	#don't let weights drop lower than 0.05,
	#as negative weights are unphysical 
	if np.any(weight_holder<0.05):
		ind=np.where(weight_holder<0.05)
		weight_holder[ind[0]]=0.05
	w=weight_holder
	return w

def momentum():
	"""
	Momentum takes into consideration sizes
	of previous steps along with results of
	gradient descent to determine the next
	step size
	"""
	step_arr=np.asarray(steps_list)
	
	if step_arr.shape[1]>1:
		step_arr=np.flip(step_arr,axis=0)
	
	if step_arr.shape[0]>5:
		last_four_steps=step_arr[:5,:]
		num_stepsy=5
	else:
		last_four_steps=step_arr
		num_stepsy=step_arr.shape[0]
	
	temp_hold=(glob_beta_matr[None,:num_stepsy]*last_four_steps.T).T
	overall_steps=np.sum(temp_hold, axis=0)
	#print("overall_steps", overall_steps)
	return overall_steps

def backprop(w):
	"""
	Returns gradient for the cost function.
	This requires calculating current CXB, and 
	numerical derivative of the change in cost function
	with the change in weight.
	"""
	t_orig_cxb=time.time()
	nabla_w = [np.zeros(w.shape)]
	#print(w)
	#here acrivation is num count/xrb 
	new_xrb_all_bins=w[None,:].T*orig_cxb#w[None,:,:].T*orig_cxb  #xrb_calculator(w)
	new_xrb=np.sum(new_xrb_all_bins,1)
	#print("New xrb:", new_xrb)
	#y - activation:
	new_xrb=new_xrb+galaxy_contri #galaxy contribution
	#print("With galaxy contri orig:", new_xrb, time.time()-t_orig_cxb)
	t_make_plot=time.time()
	#save_min_fit(new_xrb, "original_cxb_"+str(steps)+".png")
	#print("plot making time:", time.time()-t_make_plot)
	#xrb_diff_cost, xrb_diff_mcmc =[], []
	time_lsq=time.time()
	cost, xrb_diff_mcmc, num_of_training, lsqs_now, lsqs_now_full =xrb_least_squares(energyarrold, new_xrb)
	all_lsqs_now.append(lsqs_now) #all_lsqs_now.append(lsqs_now)
	all_lsqs_now_full.append(lsqs_now_full)
	#cost (equation 26: http://neuralnetworksanddeeplearning.com/chap2.html)
	cost=cost/(2.*num_of_training)#2808.
	#print("cost lsq time", cost, time.time()-time_lsq)
	t_cxb_min=time.time()
	min_cxb_now.append(new_xrb)
	t1.append(time.time())
	costs_list.append(cost)
	t_derii=time.time()
	nabla_w = xrb_calculator_derivative(w, cost)
	#print("derivative time", time.time()-t_derii)
	return nabla_w

#after making one update of the ws, we have to use the "current.npy"
def xrb_calculator_derivative(w,old_cost):
	#global weight_deriv
	#weight_deriv=w
	tderiv=time.time()
	cost=[]
	dw_metric=dw*np.identity(15)#[:,:,None]#dw*np.identity(15)#np.tile(dw*np.identity(15),parr_matr)
	dw_metric=np.tile(dw_metric,(parr_matr,1)).reshape(parr_matr,15,15)
	#print("dw_metric",dw_metric)
	altered_weight_metric=np.tile(weigh,(15,1)).reshape(15,15,parr_matr).T
	#np.tile(weigh,(15,1)).reshape(15,15,parr_matr)#np.tile(w,(15,1)).T
	#print("altered_weight_metric before addition",altered_weight_metric)
	altered_weight_metric+=dw_metric

	new_xrb_all_bins=[]
	for ele in altered_weight_metric:
		#print("altered_weight_metric ele",ele)
		new_xrb_all_bins.append(ele[None,:].T*orig_cxb)
	
	#print("orig_cxb", orig_cxb)
	new_xrb_all_bins=np.array(new_xrb_all_bins)

	new_xrb=np.sum(new_xrb_all_bins,2)

	new_xrb=new_xrb+galaxy_contri
	#print("with galaxy contri", new_xrb)
	cost, xrb_diff_mcmc, num_of_training, cdfs_deri, cdfs_full_ignore=xrb_least_squares(energyarrold, new_xrb)
	cost=cost/(2.*num_of_training)
	##### start with subtracting galaxy contribution from data ####
	#divide by delta w = 0.05 (or 0.01) to find derivative of cost function
	#equation 46 from http://neuralnetworksanddeeplearning.com/chap2.html:
	dC_dw=(cost.T-old_cost)/dw
	#dC_dw_theoretical=2*np.sum(2*orig_cxb,0)
	#print("diff in cost:", dC_dw)#, dC_dw_theoretical)
	return dC_dw

def save_min_fit(new_cxbs, name_file, plot_all=False, show_all_datasets=True):
	swiftbat_color="#4082F5"
	chandra_color="#95361D"
	rxte_color="#80ff80"
	A18_color="#BA4A00"
	plt.figure(figsize=(6, 6), dpi= 100)

	viridis = cm.get_cmap('viridis_r', len(new_cxbs))
	#print(viridis)

	counterr=0

	if show_all_datasets:
		asca_color="#C5A506"
		plt.scatter(asca['x'], asca['y'], color=asca_color, marker="s", alpha=0.5, label="$ASCA$ 1995", s=10)
		for line in asca:
			#plt.plot([line['energy'],line['energy']], [line['lower'], line['upper']], color=swiftbat_color)
			plt.plot([line['x']-line['x_err_low'],line['x']+line['x_err_up']], 
					 [line['y'], line['y']], alpha=0.5, color=asca_color)
			plt.plot([line['x'],line['x']], [line['y']-line['y_err_low'],
					 line['y']+line['y_err_up']], alpha=0.5, color=asca_color)
		
		heao1_color="#037317"
		#energy_low energy_med energy_high lower median upper
		plt.scatter(heao1['x'].data,heao1['med'].data, color=heao1_color, marker='+', label="HEAO 1999", alpha=0.5)
		for line in heao1:
			plt.plot([line['x'],line['x']], [line['med'], line['med']], color=heao1_color, alpha=0.5)
			plt.plot([line['x'],line['x']], [line['low'], line['high']], color=heao1_color, alpha=0.5)

		heao_a2_color="#0B8CAB"
		plt.scatter(heao_a2['x'], heao_a2['median'], color=heao_a2_color, marker="x", label="HEAO A4 1997", s=10, alpha=0.5)
		for line in heao_a2:
			#plt.plot([line['energy'],line['energy']], [line['lower'], line['upper']], color=swiftbat_color)
			plt.plot([line['x'],line['x']], [line['low'],line['high']], color=heao_a2_color, alpha=0.5)

		integral_color="#A46EF5"		
		plt.scatter(integral['energy_med'].data, integral['median'].data, 
					marker='+', label="$INTEGRAL$ 2007", color=integral_color, alpha=0.5)
		for line in integral:
			plt.plot([line['energy_low'],line['energy_high']], [line['median'], line['median']], color=integral_color, alpha=0.5)
			plt.plot([line['energy_med'],line['energy_med']], [line['lower'], line['upper']], color=integral_color, alpha=0.5)

	#plot errors:
	err_xs=np.linspace(0.5,110,70)
	#plt.fill_between(err_xs,xray_error_low(err_xs),xray_error_high(err_xs),color="#FBF842",alpha=0.3,label="CXB Uncertainty")
	if plot_all:
		for ele in new_cxbs:
			plt.plot(energyarrold, ele, lw=.8, ls='solid', color=viridis(counterr))#label=r"Total # "+str(counterr))
			counterr+=1
		plt.plot([],[],color=viridis(0),label="Starting model prediction")
		plt.plot([],[],color=viridis(2),label="Intermediate steps")
		plt.plot([],[],color=viridis(4),label="Final model position")
	else:
		plt.plot(energyarrold,new_cxbs[4], lw=.8, ls='solid', color=viridis(4), label="Best-fit")

	plt.scatter(chandra['energy'].data, chandra['median'].data, color=chandra_color, label=r"$Chandra$ 2016")
	#energy lower median upper
	for line in chandra:
		plt.plot([line['energy'],line['energy']], [line['lower'], line['upper']], color=chandra_color)
	
	#energy_low energy_med energy_high lower median upper
	for line in rxte:
		plt.plot([line['energy_low'],line['energy_high']], [line['median'], line['median']], color=rxte_color)
		plt.plot([line['energy_med'],line['energy_med']], [line['lower'], line['upper']], color=rxte_color)

	plt.scatter([],[], color=rxte_color, marker='+', label="RXTE 2003")

	plt.scatter(swift_bat['x'], swift_bat['y'], color=swiftbat_color, marker="s", label="$Swift$-BAT 2008", s=10)
	for line in swift_bat:
		#plt.plot([line['energy'],line['energy']], [line['lower'], line['upper']], color=swiftbat_color)
		plt.plot([line['x']-line['x_err_low'],line['x']+line['x_err_up']], 
				 [line['y'], line['y']], color=swiftbat_color)
		plt.plot([line['x'],line['x']], [line['y']-line['y_err_low'],
				 line['y']+line['y_err_up']], color=swiftbat_color)

	plt.xlabel(r"Energy [keV]",fontsize=12)
	plt.ylabel(r"E$^2$ F(E) [keV$^2$ cm$^{-2}$ s$^{-1}$ keV$^{-1}$ sr$^{-1}$]",fontsize=12)
	#plt.title("Cosmic X-ray Background")
	plt.text(2.,95,"Cosmic X-ray Background",fontsize=14)
	leg = plt.legend(fancybox=True, loc='lower right',prop={'size':9}, handlelength=3,ncol=1)#bbox_to_anchor=(1.14, 0.1))
	# set the alpha value of the legend: it will be translucent
	leg.get_frame().set_alpha(0.8)
	plt.xlim(0.56,100)#(0.5,70)#
	plt.ylim(3,120)#(3,140)#(0.03,200)#(0,65)#
	plt.yscale("log")
	plt.xscale("log")
	plt.show()
	#plt.savefig(name_file)
	plt.close()
	return


def pvals_other_datasets(new_cxbs):
	pval_asca=pvalue_dataset(asca['x'].data, asca['y'].data,
								 asca['y_err_up'].data, asca['y_err_low'].data, 
								 energyarrold, new_cxbs)
	pval_heao_a2=pvalue_dataset(heao_a2['x'].data, heao_a2['median'].data,
								 heao_a2['high'].data-heao_a2['median'].data,
								 heao_a2['median'].data-heao_a2['low'].data, 
								 energyarrold, new_cxbs)
	pval_integral=pvalue_dataset(integral['energy_med'].data, integral['median'].data,
									 integral['upper'].data-integral['median'].data, 
									 integral['median'].data-integral['lower'].data, 
									 energyarrold, new_cxbs)
	pval_heao1=pvalue_dataset(heao1['x'].data,heao1['med'].data,
								 heao1['high'].data-heao1['med'].data,heao1['med'].data-heao1['low'].data, 
								 energyarrold, new_cxbs)
	return pval_asca,pval_heao_a2,pval_heao1,pval_integral

def run_network(spec_type,
				gam,gam_sig,r,r_sig,ecut,ecut_sig,
				gam_abs=0,gam_sig_abs=0,r_abs=0,
				r_sig_abs=0,ecut_abs=0,ecut_sig_abs=0,
				gam_ctk=0,gam_sig_ctk=0,r_ctk=0,
				r_sig_ctk=0,ecut_ctk=0,ecut_sig_ctk=0):

	global parr_matr
	global weigh
	global term_step
	global steps
	global num_params
	global orig_cxb
	global t1
	global costs_list
	global steps_list
	global min_cxb_now
	global all_lsqs_now
	global all_lsqs_now_full

	#set up hyper parameters:
	#learning rate:
	parr_matr=int(output_slider_variable_parr_matr.value)#parr_matre #how many parallel routes
	term_step=int(output_slider_variable_term_step.value)#term_stepe

	steps=0
	weigh=np.random.uniform(low=0.3, high=5.0, size=15*parr_matr).reshape(15,parr_matr)
	num_params=len(weigh)

	t1=[]
	costs_list=[]
	steps_list=[]
	min_cxb_now=[]
	all_lsqs_now=[]
	all_lsqs_now_full=[]

	orig_cxb=[]

	energy_arr=np.tile(np.array([ecut,ecut_sig,r,r_sig,gam,gam_sig,1.]),23).reshape(23,7)
	energy_arr[:,6]=energyarrold
	all_energy_vals=np.tile(energy_arr,(15,1)).reshape(15,23,7)

	if spec_type>1:
		energy_arr_ctn=np.tile(np.array([ecut_abs,ecut_sig_abs,
										 r_abs,r_sig_abs,gam_abs,
										 gam_sig_abs,1.]),23).reshape(23,7)
		energy_arr_ctn[:,6]=energyarrold
		all_energy_vals[5:,:,:]=np.tile(energy_arr_ctn,(10,1)).reshape(10,23,7)
	if spec_type==3:
		energy_arr_ctk=np.tile(np.array([ecut_ctk,ecut_sig_ctk,
										 r_ctk,r_sig_ctk,gam_ctk,
										 gam_sig_ctk,1.]),23).reshape(23,7)
		energy_arr_ctk[:,6]=energyarrold
		all_energy_vals[10:,:,:]=np.tile(energy_arr_ctk,(5,1)).reshape(5,23,7)
		

	for i in range(0,15):
		blah=interpolator_list[i](all_energy_vals[i])
		orig_cxb.append(blah)

	orig_cxb=np.array(orig_cxb)
	#print("orig_cxb",orig_cxb)

	while steps<=term_step:#promising<20:
		#print("STEP:", steps)
		weigh=update_weights(glob_eta, weigh)#update_weights(0.1, weigh)
		#print("Costs list:", costs_list)
		if steps>31:
			#print("costs diff:", np.diff(np.array(costs_list)[-30:]))
			if np.any(np.array(costs_list)[-1,:]<2.0):
				break
		steps+=1

	#print("total time:", time.time()-t_this_run)
	find_min=np.argmin(np.array(costs_list)[-1])
	#print("LSQ final",np.array(all_lsqs_now)[-1,:,find_min])

	
	find_cdf=np.array(all_lsqs_now)[-1,:,find_min]
	find_cdf_full=np.array(all_lsqs_now_full)[-1,:,find_min]
	#print("find cdf",find_cdf)
	#cdf_val=stats.chi2.cdf(find_cdf, 56)
	right_ones=np.linspace(0,steps-1,num_of_fig_lines)
	right_ones=right_ones.astype(int)
	num_steps=len(min_cxb_now)
	min_cxb_now=np.array(min_cxb_now).reshape(num_steps,parr_matr,len(energyarrold))
	pval_asca,pval_heao_a2,pval_heao1,pval_integral=pvals_other_datasets(min_cxb_now[-1,find_min,:])
	print("")
	cdf_val=np.array([stats.chi2.cdf(find_cdf[0], 18),
					  stats.chi2.cdf(find_cdf[1], 19),
					  stats.chi2.cdf(find_cdf[2], 18)])
	cdf_val_full=stats.chi2.cdf(find_cdf_full, 18)
	pval=1.-cdf_val
	#print('\033[1m'+"Error Set A")
	print('\033[0m'+"Probability with respect to errors in each data set (shown in figure with error bars):")
	print("Chandra:", pval[0])
	print("ASCA:", pval_asca)
	print("RXTE:", pval[1])
	print("Swift-BAT:", pval[2])
	print("INTEGRAL:", pval_integral)
	print("HEAO (1999):", pval_heao1)
	print("HEAO-A4 (1997):", pval_heao_a2)
	print("")
	print("Comments:")
	accept=False
	if pval[0]>=0.003 or pval_asca>=0.003:
		print("Agrees within 3σ significance level with at least one CXB measurement at low energies.")
		accept=True
	else:
		print("Disagrees with all data sets at low energies")

	if (pval[1]>=0.003 and pval[2]>=0.003) or pval_integral>=0.003:
		print("Agrees within 3σ significance level with at least one CXB measurement at intermediate and high energies.")
	else:
		accept=False
		print("Disagrees with all data sets at high energies.")

	'''
	if np.min(pval)<0.003:
		print('\033[1m' +"Rejected as this parameter set falls outside 3σ significance \nlevel with respect to at least one dataset.")
	else:
		print('\033[1m' +"Accepted as this parameter set is within 3σ significance level \nwith respect to all three datasets.")
		print("")
	
	pval_full=1.-cdf_val_full
	print("")
	print('\033[1m'+"Error Set B")
	print('\033[0m'+"Probability with respect to full range of CXB errors (shown in figure with shaded yellow region):")
	print("Chandra data set:", pval_full[0])
	print("RXTE data set:", pval_full[1])
	print("Swift-BAT data set:", pval_full[2])
	if np.min(pval_full)>=0.317:
		print("With respect to full range of CXB errors, \nthis region of the parameter space falls within \n1σ significance level of all data sets.")
	elif np.min(pval_full)>=0.05:
		print("With respect to full range of CXB errors, \nthis region of the parameter space falls within \n2σ significance level of all data sets.")
	elif np.min(pval_full)>=0.003:
		print("With respect to full range of CXB errors, \nthis region of the parameter space falls within \n3σ significance level of all data sets.")
	else:
		print("This parameter set falls outside 3σ significance level with \nrespect to at least one data set.")
	print("")
	'''
	if accept:
		print('\033[1m'+"Result: Acceptable") 
	else:
		print('\033[1m'+"Result: Rejected")
	print("")
	name_conv=str(gam)+"_"+str(gam_sig)+"_"+str(ecut)+"_"+str(r)
	save_min_fit(min_cxb_now[right_ones,find_min,:],"final_result_"+name_conv+"_"+str(term_step)+".png")
	plt.figure()
	plt.plot(np.array(costs_list)[:,find_min])
	plt.xlabel("steps")
	plt.ylabel("cost")
	plt.yscale("log")
	plt.title("Lowest cost: "+str(round(np.min(np.array(costs_list)[-1]),2)))
	print("Convergence of the best fit model prediction:")
	plt.show()
	
output_slider_variable_parr_matr=widgets.Text()
output_slider_variable_term_step=widgets.Text()

output_slider_variable_R=widgets.Text()
output_slider_variable_E=widgets.Text()
output_slider_variable_G=widgets.Text()
output_slider_variable_Gstd=widgets.Text()
output_slider_variable_Rstd=widgets.Text()
output_slider_variable_Estd=widgets.Text()

output_slider_variable_R_ctn=widgets.Text()
output_slider_variable_E_ctn=widgets.Text()
output_slider_variable_G_ctn=widgets.Text()
output_slider_variable_Gstd_ctn=widgets.Text()
output_slider_variable_Rstd_ctn=widgets.Text()
output_slider_variable_Estd_ctn=widgets.Text()

output_slider_variable_R_ctk=widgets.Text()
output_slider_variable_E_ctk=widgets.Text()
output_slider_variable_G_ctk=widgets.Text()
output_slider_variable_Gstd_ctk=widgets.Text()
output_slider_variable_Rstd_ctk=widgets.Text()
output_slider_variable_Estd_ctk=widgets.Text()

spec_type=widgets.Text()

def f_start(Spectra_type):
    spec_type.value = str(Spectra_type)
    
options_spec=['Same params for all absorption bins', 
              'Different params for absorbed and unabosorbed AGN', 
              'Different params for unasorbed, Compton-thin and Compton-thick']


def initialize():
	a=interact(f_start,
    	Spectra_type=[(options_spec[0], 1), 
            	 (options_spec[1], 2), 
            	 (options_spec[2], 3)],
	    value=1,
	    description='Parameter set specification:',
	)



def f(Threads=20,Steps=8000):
    output_slider_variable_parr_matr.value = str(Threads)
    output_slider_variable_term_step.value = str(Steps)

    
def f1(Γ,σ_Γ,R,σ_R,Ecut,σ_E):
    output_slider_variable_G.value = str(Γ)
    output_slider_variable_Gstd.value = str(σ_Γ)
    output_slider_variable_R.value = str(R)
    output_slider_variable_Rstd.value = str(σ_R)
    output_slider_variable_E.value = str(Ecut)
    output_slider_variable_Estd.value = str(σ_E)
    
def f2(Γ,σ_Γ,R,σ_R,Ecut,σ_E,
       Γ_abs,σ_Γ_abs,
       R_abs,σ_R_abs,
       Ecut_abs,σ_E_abs):
    output_slider_variable_G.value = str(Γ)
    output_slider_variable_Gstd.value = str(σ_Γ)
    output_slider_variable_R.value = str(R)
    output_slider_variable_Rstd.value = str(σ_R)
    output_slider_variable_E.value = str(Ecut)
    output_slider_variable_Estd.value = str(σ_E)
    output_slider_variable_G_ctn.value = str(Γ_abs)
    output_slider_variable_Gstd_ctn.value = str(σ_Γ_abs)
    output_slider_variable_R_ctn.value = str(R_abs)
    output_slider_variable_Rstd_ctn.value = str(σ_R_abs)
    output_slider_variable_E_ctn.value = str(Ecut_abs)
    output_slider_variable_Estd_ctn.value = str(σ_E_abs)
    
def f3(Γ,σ_Γ,R,σ_R,Ecut,σ_E,
       Γ_ctn,σ_Γ_ctn,R_ctn,σ_R_ctn,
       Ecut_ctn,σ_E_ctn,Γ_ctk,σ_Γ_ctk,
       R_ctk,σ_R_ctk,Ecut_ctk,σ_E_ctk):
    output_slider_variable_G.value = str(Γ)
    output_slider_variable_Gstd.value = str(σ_Γ)
    output_slider_variable_R.value = str(R)
    output_slider_variable_Rstd.value = str(σ_R)
    output_slider_variable_E.value = str(Ecut)
    output_slider_variable_Estd.value = str(σ_E)
    output_slider_variable_G_ctn.value = str(Γ_ctn)
    output_slider_variable_Gstd_ctn.value = str(σ_Γ_ctn)
    output_slider_variable_R_ctn.value = str(R_ctn)
    output_slider_variable_Rstd_ctn.value = str(σ_R_ctn)
    output_slider_variable_E_ctn.value = str(Ecut_ctn)
    output_slider_variable_Estd_ctn.value = str(σ_E_ctn)
    output_slider_variable_G_ctk.value = str(Γ_ctk)
    output_slider_variable_Gstd_ctk.value = str(σ_Γ_ctk)
    output_slider_variable_R_ctk.value = str(R_ctk)
    output_slider_variable_Rstd_ctk.value = str(σ_R_ctk)
    output_slider_variable_E_ctk.value = str(Ecut_ctk)
    output_slider_variable_Estd_ctk.value = str(σ_E_ctk)        
    

def make_sliders():
	c=interact(f,
           Threads=(2,100,1),
           Steps=(1000,10000,1000))

	print("")
	print("")

	if spec_type.value=="1":
		b=interact(f1,
					Γ=(1.4,2.2,0.01),
					σ_Γ=(0.0,0.5,0.01),
					R=(0,2.0,0.01),
					σ_R=(0.0,1.0,0.01),
					Ecut=(50.,490.,1),
					σ_E=(0.0,100,1))
	elif spec_type.value=="2":
	    b=interact(f2,
					Γ=(1.4,2.2,0.01),
					σ_Γ=(0.0,0.5,0.01),
					R=(0,2.0,0.01),
					σ_R=(0.0,1.0,0.01),
					Ecut=(50.,490.,1),
					σ_E=(0.0,100,1),
					Γ_abs=(1.4,2.2,0.01),
					σ_Γ_abs=(0.0,0.5,0.01),
					R_abs=(0,2.0,0.01),
					σ_R_abs=(0.0,1.0,0.01),
					Ecut_abs=(50.,490.,1),
					σ_E_abs=(0.0,100,1))
	elif spec_type.value=="3":
	    b=interact(f3,
					Γ=(1.4,2.2,0.01),
					σ_Γ=(0.0,0.5,0.01),
					R=(0,2.0,0.01),
					σ_R=(0.0,1.0,0.01),
					Ecut=(50.,490.,1),
					σ_E=(0.0,100,1),
					Γ_ctn=(1.4,2.2,0.01),
					σ_Γ_ctn=(0.0,0.5,0.01),
					R_ctn=(0,2.0,0.01),
					σ_R_ctn=(0.0,1.0,0.01),
					Ecut_ctn=(50.,490.,1),
					σ_E_ctn=(0.0,100,1),
					Γ_ctk=(1.4,2.2,0.01),
					σ_Γ_ctk=(0.0,0.5,0.01),
					R_ctk=(0,2.0,0.01),
					σ_R_ctk=(0.0,1.0,0.01),
					Ecut_ctk=(50.,490.,1),
					σ_E_ctk=(0.0,100,1))

def execute():
	Rval=float(output_slider_variable_R.value)
	Eval=float(output_slider_variable_E.value)
	Gval=float(output_slider_variable_G.value)
	Rstdval=float(output_slider_variable_Rstd.value)
	Estdval=float(output_slider_variable_Estd.value)
	Gstdval=float(output_slider_variable_Gstd.value)
	print("Unabsorbed:","<R>: "+str(Rval)+", σ_R: "+str(Rstdval)+\
		  ", <Γ>: "+str(Gval)+", σ_Γ: "+str(Gstdval)+", <E>: "+str(Eval)+", σ_E:"+str(Estdval))
	if spec_type.value=="1":
		if abs(36-Estdval)<1 and abs(0.14-Rstdval)<0.005 and Gstdval>=0.1:
			#print("Total probability distribution:")
			just_stats(Rval,Eval,Gstdval,Gval)
			one_fig(Rval,Eval,Gstdval,Gval)
		else:
			run_network(float(spec_type.value),
						Gval,Gstdval,Rval,Rstdval,Eval,Estdval)
	else:
		Rval_ctn=float(output_slider_variable_R_ctn.value)
		Eval_ctn=float(output_slider_variable_E_ctn.value)
		Gval_ctn=float(output_slider_variable_G_ctn.value)
		Rstdval_ctn=float(output_slider_variable_Rstd_ctn.value)
		Estdval_ctn=float(output_slider_variable_Estd_ctn.value)
		Gstdval_ctn=float(output_slider_variable_Gstd_ctn.value)
		if spec_type.value=="2":
			print("Absorbed:","<R>: "+str(Rval_ctn)+", σ_R: "+str(Rstdval_ctn)+\
				  ", <Γ>: "+str(Gval_ctn)+", σ_Γ: "+str(Gstdval_ctn)+", <E>: "+str(Eval_ctn)+\
				  ", σ_E: "+str(Estdval_ctn))
			run_network(float(spec_type.value),
						Gval,Gstdval,Rval,Rstdval,Eval,Estdval,
						Gval_ctn,Gstdval_ctn,Rval_ctn,
 						Rstdval_ctn,Eval_ctn,Estdval_ctn)   
		else:
			print("Compton-thin:","<R>: "+str(Rval_ctn)+", σ_R: "+str(Rstdval_ctn)+\
				  ", <Γ>: "+str(Gval_ctn)+", σ_Γ: "+str(Gstdval_ctn)+", <E>: "+str(Eval_ctn)+\
				  ", σ_E: "+str(Estdval_ctn))
			print("Compton-thick:","<R>: "+str(Rval_ctk)+", σ_R: "+str(Rstdval_ctk)+\
				  ", <Γ>: "+str(Gval_ctk)+", σ_Γ: "+str(Gstdval_ctk)+", <E>: "+str(Eval_ctk)+\
				  ", σ_E: "+str(Estdval_ctk))
			Rval_ctk=float(output_slider_variable_R_ctk.value)
			Eval_ctk=float(output_slider_variable_E_ctk.value)
			Gval_ctk=float(output_slider_variable_G_ctk.value)
			Rstdval_ctk=float(output_slider_variable_Rstd_ctk.value)
			Estdval_ctk=float(output_slider_variable_Estd_ctk.value)
			Gstdval_ctk=float(output_slider_variable_Gstd_ctk.value)
			run_network(float(spec_type.value),
				    Gval,Gstdval,Rval,Rstdval,Eval,Estdval,
                    Gval_ctn,Gstdval_ctn,Rval_ctn,
                    Rstdval_ctn,Eval_ctn,Estdval_ctn,
                    Gval_ctk,Gstdval_ctk,Rval_ctk,
                    Rstdval_ctk,Eval_ctk,Estdval_ctk)

dataset_names_latex=[r"$\it{Chandra}$",r"$\it{RXTE}$",r"$\it{Swift}$-BAT","Overall"]
dataset_names=["Chandra","RXTE","Swift-BAT"]
dataset_energy_range=["E < 8 keV","3 keV < E < 20 keV", "E > 15 keV"]
pval_func_c,pval_func_r,pval_func_s,pval_func=cdf_func_total()

def one_fig(R_fix,Ecut_sample,std,gam_sample):

	point_dens=80
	vmin=0
	vmax=1

	fig, ax = plt.subplots(ncols=2, nrows=3, sharey=False, sharex=False,
						   figsize=(10, 10), dpi= 80, facecolor='w', 
						   edgecolor='k')
	plt.xticks(fontsize=1)#), rotation=90)
	plt.yticks(fontsize=1)#, rotation=90)

	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,R_fix).reshape(point_dens*point_dens,4)
	fake_large[:,1]=std
	ecut_links=np.linspace(50.,490.,point_dens)
	gamma_links=np.linspace(1.4,2.2,point_dens)

	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,2]=ecut_links  
		fake_large[i*point_dens:(i+1)*point_dens,0]=gamma_links[i]

	cost_vals=pval_func(fake_large)

	cm = plt.cm.get_cmap('viridis')

	X, Y=np.meshgrid(ecut_links,gamma_links)

	ax[0,0].scatter(fake_large[:,2],fake_large[:,0],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)
	
	ax[0,0].scatter([Ecut_sample],[gam_sample],marker="X",color="k",s=200)
	ax[0,0].scatter([Ecut_sample],[gam_sample],marker="x",color="white",s=50)

	rel_pval=pval_func([gam_sample,std,Ecut_sample,R_fix])
	
	ax[0,0].set_xlabel(r"$\langle E_{\rm cutoff} \rangle$",fontsize=16)
	ax[0,0].set_ylabel(r"$\langle \Gamma \rangle$",fontsize=16)
	#ax[0,0].set_aspect(aspects[0])
	ax[0,0].set_xlim(50,500)
	ax[0,0].set_ylim(1.4,2.2)
	ax[0,0].tick_params(labelsize=14)

	#std-R axis
	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,Ecut_sample).reshape(point_dens*point_dens,4)
	fake_large[:,0]=gam_sample
	r_links=np.linspace(0,2.0,point_dens)
	std_links=np.linspace(0.1,0.5,point_dens)

	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,3]=r_links
		fake_large[i*point_dens:(i+1)*point_dens,1]=std_links[i]

	cost_vals=pval_func(fake_large)

	a=ax[0,1].scatter(fake_large[:,3],fake_large[:,1],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)

	ax[0,1].scatter([R_fix],[std],marker="X",color="k",s=200)
	ax[0,1].scatter([R_fix],[std],marker="x",color="white",s=50)


	ax[0,1].set_xlabel(r"$\langle R \rangle$",fontsize=16)
	ax[0,1].set_ylabel(r"$\sigma_{\Gamma}$",fontsize=16)
	#ax[0,1].set_aspect(aspects[1])
	ax[0,1].set_xlim(0,1.5)
	ax[0,1].set_ylim(0.1,0.5)
	ax[0,1].tick_params(labelsize=14)

	#R-Gamma axis
	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,Ecut_sample).reshape(point_dens*point_dens,4)
	fake_large[:,1]=std
	gam_links=np.linspace(1.4,2.2,point_dens)
	r_links=np.linspace(0,2.0,point_dens)


	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,0]=gam_links
		fake_large[i*point_dens:(i+1)*point_dens,3]=r_links[i]

	    
	cost_vals=pval_func(fake_large)

	cm = plt.cm.get_cmap('viridis')

	ax[1,0].scatter(fake_large[:,0],fake_large[:,3],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)

	ax[1,0].scatter([gam_sample],[R_fix],marker="X",color="k",s=200)
	ax[1,0].scatter([gam_sample],[R_fix],marker="x",color="white",s=50)
	
	ax[1,0].set_xlabel(r"$\langle \Gamma \rangle$",fontsize=16)
	ax[1,0].set_ylabel(r"$\langle R \rangle$",fontsize=16)
	#ax[1,1].set_aspect(aspects[3])
	ax[1,0].set_xlim(1.4,2.2)
	ax[1,0].set_ylim(0,1.5)
	ax[1,0].tick_params(labelsize=14)

	#std-Gamma axis
	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,Ecut_sample).reshape(point_dens*point_dens,4)
	fake_large[:,3]=R_fix
	gam_links=np.linspace(1.4,2.2,point_dens)
	std_links=np.linspace(0.1,0.5,point_dens)


	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,0]=gam_links
		fake_large[i*point_dens:(i+1)*point_dens,1]=std_links[i]

	    
	cost_vals=pval_func(fake_large)

	cm = plt.cm.get_cmap('viridis')

	a=ax[1,1].scatter(fake_large[:,0],fake_large[:,1],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)

	ax[1,1].scatter([gam_sample],[std],marker="X",color="k",s=200)
	ax[1,1].scatter([gam_sample],[std],marker="x",color="white",s=50)
	
	ax[1,1].set_xlabel(r"$\langle \Gamma \rangle$",fontsize=16)
	ax[1,1].set_ylabel(r"$\sigma_{\Gamma}$",fontsize=16)
	#ax[1,0].set_aspect(aspects[2])
	ax[1,1].set_xlim(1.4,2.2)
	ax[1,1].set_ylim(0.1,0.5)
	ax[1,1].tick_params(labelsize=14)

	#R-Ecut axis
	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,gam_sample).reshape(point_dens*point_dens,4)
	fake_large[:,1]=std
	gam_links=np.linspace(1.4,2.2,point_dens)
	r_links=np.linspace(0,2.0,point_dens)


	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,2]=ecut_links  
		fake_large[i*point_dens:(i+1)*point_dens,3]=r_links[i]

	cost_vals=pval_func(fake_large)

	cm = plt.cm.get_cmap('viridis')

	ax[2,0].scatter(fake_large[:,2],fake_large[:,3],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)

	ax[2,0].scatter([Ecut_sample],[R_fix],marker="X",color="k",s=200)
	ax[2,0].scatter([Ecut_sample],[R_fix],marker="x",color="white",s=50)
	
	ax[2,0].set_xlabel(r"$\langle E_{\rm cutoff} \rangle$",fontsize=16)
	ax[2,0].set_ylabel(r"$\langle R \rangle$",fontsize=16)
	#ax[2,1].set_aspect(aspects[5])
	ax[2,0].set_xlim(50.,500.)
	ax[2,0].set_ylim(0,1.5)
	ax[2,0].tick_params(labelsize=14)

	#std-Ecut axis
	#pval_func([gam_sample,std,Ecut_sample,R_fix])
	fake_large=np.full(4*point_dens**2,gam_sample).reshape(point_dens*point_dens,4)
	fake_large[:,3]=R_fix
	gam_links=np.linspace(1.4,2.2,point_dens)
	std_links=np.linspace(0.1,0.5,point_dens)


	for i in range(0,point_dens):
		fake_large[i*point_dens:(i+1)*point_dens,2]=ecut_links
		fake_large[i*point_dens:(i+1)*point_dens,1]=std_links[i]

	cost_vals=pval_func(fake_large)

	cm = plt.cm.get_cmap('viridis')

	ax[2,1].scatter(fake_large[:,2],fake_large[:,1],vmin=vmin, vmax=vmax,c=1.-cost_vals, s=50,cmap=cm)

	ax[2,1].scatter([Ecut_sample],[std],marker="X",color="k",s=200)
	ax[2,1].scatter([Ecut_sample],[std],marker="x",color="white",s=50)
	
	ax[2,1].set_xlabel(r"$\langle E_{\rm cutoff} \rangle$",fontsize=16)
	ax[2,1].set_ylabel(r"$\sigma_{\Gamma}$",fontsize=16)
	#ax[20].set_aspect(aspects[4])
	ax[2,1].set_xlim(50.,500.)
	ax[2,1].set_ylim(0.1,0.5)
	ax[2,1].tick_params(labelsize=14)

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.2, hspace=0.3)
	fig.subplots_adjust(right=0.9,top=0.92)
	#cbar_ax = fig.add_axes([0.81, 0.15, 0.05, 0.2])
	cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
	clb=fig.colorbar(a, cax=cbar_ax)
	clb.ax.set_title(r'P($\chi^2$)')
	fig.suptitle(dataset_names_latex[-1]+": \n"+r"$\langle R \rangle$: "+str(round(R_fix,2))+",     "+\
			  r"$\langle E_{\rm cutoff} \rangle$: "+str(round(Ecut_sample))+",     "+\
			  r"$\langle \Gamma \rangle$: "+str(round(gam_sample,2))+",     "+\
			  r"$\sigma_\Gamma$: "+str(round(std,2)),
			  fontsize=18, x=0.5, y=0.99)
	plt.show()
	return

def just_stats(R_fix,Ecut_sample,std,gam_sample):
	indi=0
	bool_vals=[]

	pval_funcs=[pval_func_c,pval_func_r,pval_func_s]

	#print("Null hypothesis: This region of parameter space can fit the CXB.\n")
	print("Probability for:")

	for pval_func_now in pval_funcs:
		rel_pval=pval_func_now([gam_sample,std,Ecut_sample,R_fix])
		print(dataset_names[indi]+": "+str(1.-rel_pval[0]))
		if 1.-rel_pval[0]>0.003:#>0.683: #this is really the CDF value
			bool_vals.append(True)
		else:
			bool_vals.append(False)
		indi+=1

	if np.all(np.array(bool_vals)):
		print('\033[1m'+"Accepted as this region of the parameter space falls within \nacceptable significance level with respect to all data sets.")
	else:
		print('\033[1m'+"Rejected as this spectral set falls at > 3σ significance level \nwith respect to at least one data set.")
	return
'''
def get_new_vals(param_option):
	step_num=output_slider_variable_term_step.value
	num_threads=output_slider_variable_parr_matr.value

	Rval=float(output_slider_variable_R.value)
	Eval=float(output_slider_variable_E.value)
	Gval=float(output_slider_variable_G.value)
	Rstdval=float(output_slider_variable_Rstd.value)
	Estdval=float(output_slider_variable_Estd.value)
	Gstdval=float(output_slider_variable_Gstd.value)
	if param_option=="1":
		return step_num,num_threads,Rval,\
				Eval,Gval,Rstdval,Estdval,Gstdval
	if param_option!="1":
		Rval_ctn=float(output_slider_variable_R_ctn.value)
		Eval_ctn=float(output_slider_variable_E_ctn.value)
		Gval_ctn=float(output_slider_variable_G_ctn.value)
		Rstdval_ctn=float(output_slider_variable_Rstd_ctn.value)
		Estdval_ctn=float(output_slider_variable_Estd_ctn.value)
		Gstdval_ctn=float(output_slider_variable_Gstd_ctn.value)
		if param_option=="2":
			return step_num,num_threads,Rval,\
				Eval,Gval,Rstdval,Estdval,Gstdval,\
				Rval_ctn, Eval_ctn,Gval_ctn,Rstdval_ctn,
		else: param_option=="3":
			Rval_ctk=float(output_slider_variable_R_ctk.value)
			Eval_ctk=float(output_slider_variable_E_ctk.value)
			Gval_ctk=float(output_slider_variable_G_ctk.value)
			Rstdval_ctk=float(output_slider_variable_Rstd_ctk.value)
			Estdval_ctk=float(output_slider_variable_Estd_ctk.value)
			Gstdval=float(output_slider_variable_Gstd_ctk.value)
			return step_num,num_threads,Rval,\
					Eval,Gval,Rstdval,Estdval,Gstdval,\
					Rval_ctn, Eval_ctn,Gval_ctn,Rstdval_ctn,\
					Estdval_ctn,Gstdval_ctn,Rval_ctk, Eval_ctk,
					Gval_ctk,Rstdval_ctk,Estdval_ctk,Gstdval_ctk
'''
