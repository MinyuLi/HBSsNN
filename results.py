import zipfile
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from matplotlib.ticker import MultipleLocator
import tensorflow as tf
from tensorflow.keras.models import load_model


rcParams['font.size'] = 12
olddpi = rcParams["savefig.dpi"]
print(f"olddpi={olddpi}")
rcParams["savefig.dpi"] = 240


#unzip the sample file.
zipped_files = [f for f in os.listdir('sample') if f.endswith('.zip')]
for z in zipped_files:
	zip_path = os.path.join('sample', z)
	with zipfile.ZipFile(zip_path, 'r') as zip_ref:
		zip_ref.extractall('sample')



gruNN = load_model(os.path.join('.', 'model', 'GRU.h5'))
lstmNN = load_model(os.path.join('.', 'model', 'LSTM.h5'))

def plot_predict_rst(axs, nn, sample_file):
	sampel_path = os.path.join('.', 'sample', sample_file+'.sp')
	samples = np.loadtxt(sampel_path, dtype="float", unpack=True, comments=r'#', ndmin=2)
	samples = np.array(samples.T)

	# Select columns from the 9th column to the last column
	selected_columns = samples[:, 9:109]
	
	# Create a boolean mask where rows with all zeros in the selected columns are False
	mask = ~(np.all(selected_columns == 0, axis=1))
	
	# Filter train_samples using the mask
	samples = samples[mask]
	
	X_test = samples[:,9:109]
	e1 = samples[:,1:2]
	is_teo = samples[:,8:9]
	
	# Normalize input
	row_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
	X_test = X_test / row_norms

	e2 = nn.predict(X_test)

	axs.plot([0., 1.0], [0., 1.0], "k",   alpha=0.3)
	axs.plot([0., 1.0], [0.15, 1.15], "k--",alpha=0.3)
	axs.plot([0., 1.0], [-0.15, 0.85],"k--",alpha=0.3)
	axs.set_xlim(0., 1.0)
	axs.set_ylim(0., 1.0)
	no_teo_mask = np.where(is_teo == 0)
	e1_no_teo = e1[no_teo_mask]
	e2_no_teo = e2[no_teo_mask]
	axs.scatter(e1_no_teo, e2_no_teo, c='k', s=10, marker='o', alpha=0.5)
	teo_mask = np.where(is_teo == 1)
	e1_teo = e1[teo_mask]
	e2_teo = e2[teo_mask]
	axs.scatter(e1_teo, e2_teo, c="r", s=10, marker='o', alpha=0.6)
	axs.xaxis.set_major_locator(MultipleLocator(0.2))
	axs.yaxis.set_major_locator(MultipleLocator(0.2))
	
	acc = (np.sum( np.abs(e1-e2)<0.15).astype(np.float64) / (len(e1)))
	axs.set_xlabel("accuracy={0:.2f}%".format(acc*100))


fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
axes1, axes2, axes3, axes4, axes5, axes6, axes7, axes8 = axs.flat

axes1.set_ylabel("GRU", fontsize=13)
axes5.set_ylabel("LSTM", fontsize=13)
axes1.set_title("OGLE HBSs", fontsize=13)
axes2.set_title("Kepler HBSs", fontsize=13)
axes3.set_title("TESS HBSs", fontsize=13)
axes4.set_title("Eccentric binaries", fontsize=13)

plot_predict_rst(axes1, gruNN, 'OGLE')
plot_predict_rst(axes2, gruNN, 'Kepler')
plot_predict_rst(axes3, gruNN, 'TESS')
plot_predict_rst(axes4, gruNN, 'EccBs')
plot_predict_rst(axes5, lstmNN, 'OGLE')
plot_predict_rst(axes6, lstmNN, 'Kepler')
plot_predict_rst(axes7, lstmNN, 'TESS')
plot_predict_rst(axes8, lstmNN, 'EccBs')

plt.tight_layout()
plt.show()


