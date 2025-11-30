import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl

from TorchDiffEqPack.odesolver import odesolve

import time

import matplotlib.pyplot as plt

import numpy as np

class ChunkedvalDataset(Dataset):
	def __init__(self, data, chunk_size=10, skips = 1):
		
		self.chunk_size = chunk_size
		self.total_time_steps = data["states"].size(-1)
		self.start_indices = list(range(self.total_time_steps - self.chunk_size + 1))[::skips]

		self.pod_data = torch.load("pod_freefem_pinball_30.pt")

		self.data = self.project_data(data)

	def project_data(self, data):

		states = data["states"]
		
		projected_states = torch.matmul(self.pod_data["modes"][:,:100].T, torch.matmul(self.pod_data["M"],states))
		
		projected_data = {
			"states": projected_states,
			"t": data["t"]
		}
		return projected_data

	def __len__(self):
		return len(self.start_indices)

	def __getitem__(self, index):
		start_idx = self.start_indices[index]
		end_idx = start_idx + self.chunk_size
		chunk = {}
		for key in self.data:
			if self.data[key].ndim == 1:  # 't' has shape [nb_timesteps]
				chunk[key] = self.data[key][start_idx:end_idx]
			else:  # 'states' has shape [nbDOFs, nb_timesteps]
				chunk[key] = self.data[key][:, start_idx:end_idx]
		return chunk

	def update_chunk_size(self, new_chunk_size, new_skips):
		self.chunk_size = new_chunk_size
		self.start_indices = list(range(self.total_time_steps - self.chunk_size + 1))[::new_skips]

# DataModule class
class CustomDataModule(pl.LightningDataModule):
	def __init__(self, val_dataset, batch_size=128, num_workers=4):
		super().__init__()
		self.val_dataset = val_dataset
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage=None):
		# You can further split the train dataset into train and validation if needed
		if stage == 'fit' or stage is None:
			self.val_loader = DataLoader(self.val_dataset, batch_size=128, num_workers=self.num_workers, pin_memory=True)

	def train_dataloader(self):
		return self.val_loader

	def val_dataloader(self):
		return self.val_loader

	def test_dataloader(self):
		return self.val_loader

class DerivativeEstimator(nn.Module):
	def __init__(self, redamat, device, N_tensor , dt=0.02, tronc=10):
		super().__init__()
		self.is_augmented = is_augmented
		self.is_phy = is_phy
		self.device = device

		# ---- choose ONE dtype for all math (float32 for speed, or float64 for parity) ----
		dty = torch.float32  # or torch.float64

		self.register_buffer("dt", torch.tensor(dt, device=self.device, dtype=dty))
		self.register_buffer("dt_skip", torch.tensor(dt * 20, device=self.device, dtype=dty))
		self.register_buffer("Ap", torch.tensor(redamat, device=self.device, dtype=dty))

		hr_data = torch.load("hr_freefem_pinball_30.pt", map_location=self.device)

		# cast HR matrices to the same dtype + device
		def _buf(name):
			self.register_buffer(name, torch.tensor(hr_data[name], device=self.device, dtype=dty))

		_buf("PHIu_pod_red_u")
		_buf("PHIv_pod_red_u")
		_buf("dPHIudx_pod_red_u")
		_buf("dPHIudy_pod_red_u")
		_buf("pod_t_PHIu_t_Wu_red_u")
		_buf("PHIu_pod_red_v")
		_buf("PHIv_pod_red_v")
		_buf("dPHIvdx_pod_red_v")
		_buf("dPHIvdy_pod_red_v")
		_buf("pod_t_PHIv_t_Wu_red_v")

		self.size  = int(size)
		self.tronc = int(tronc)
		p = self.tronc
		dt = self.dt  # tensor scalar

		# ---- sizes for U and V partitions (may differ) ----
		self.Ng_u = int(self.PHIu_pod_red_u.shape[0])  # e.g., 190
		self.Ng_v = int(self.PHIu_pod_red_v.shape[0])  # e.g., 196

		# ---- build concatenated weights for fast RHS (separate U and V) ----
		# each PHI*_u is (Ng_u, p) → stack to (4*Ng_u, p)
		W_all_u = torch.cat([
			self.PHIu_pod_red_u,  self.PHIv_pod_red_u,
			self.dPHIudx_pod_red_u, self.dPHIudy_pod_red_u
		], dim=0).contiguous()  # (4*Ng_u, p)

		# each PHI*_v is (Ng_v, p) → stack to (4*Ng_v, p)
		W_all_v = torch.cat([
			self.PHIu_pod_red_v,  self.PHIv_pod_red_v,
			self.dPHIvdx_pod_red_v, self.dPHIvdy_pod_red_v
		], dim=0).contiguous()  # (4*Ng_v, p)

		# projections are (20,Ng_u) and (20,Ng_v) → concat horizontally to (20, Ng_u+Ng_v)
		P_proj = torch.cat([
			self.pod_t_PHIu_t_Wu_red_u,   # (20, Ng_u)
			self.pod_t_PHIv_t_Wu_red_v    # (20, Ng_v)
		], dim=1).contiguous()            # (20, Ng_u+Ng_v)

		self.register_buffer("W_all_u", W_all_u, persistent=True)   # used in rhs()
		self.register_buffer("W_all_v", W_all_v, persistent=True)   # used in rhs()
		self.register_buffer("P_proj",   P_proj,   persistent=True) # used in rhs()

		# ---- use only the top-left p×p block of Ap when building implicit matrices ----
		Ap_p = self.Ap[:p, :p]
		eye_p = torch.eye(p, device=self.device, dtype=dty)
		mat_1st = eye_p - dt * Ap_p
		mat_2nd = 3 * eye_p - 2 * dt * Ap_p

		LU_1st, pivots_1st = torch.linalg.lu_factor(mat_1st)
		LU_2nd, pivots_2nd = torch.linalg.lu_factor(mat_2nd)

		self.register_buffer("LU_1st", LU_1st)
		self.register_buffer("LU_2nd", LU_2nd)
		self.register_buffer("pivots_1st", pivots_1st)
		self.register_buffer("pivots_2nd", pivots_2nd)

	# @torch.jit.export
	def rhs(self, a: torch.Tensor) -> torch.Tensor:
		"""
		Batched HR RHS.
		a: (B, p)
		returns: (B, p)
		"""
		# --- U block ---
		zu = torch.nn.functional.linear(a, self.W_all_u)  # (B, 4*Ng_u)
		uu, vu, dxuu, dyuu = zu.split(self.Ng_u, dim=1)   # each (B, Ng_u)
		fu = uu.mul(dxuu)
		fu = torch.addcmul(fu, vu, dyuu)                  # fu = uu*dxuu + vu*dyuu, (B, Ng_u)

		# --- V block ---
		zv = torch.nn.functional.linear(a, self.W_all_v)  # (B, 4*Ng_v)
		uv, vv, dxvv, dyvv = zv.split(self.Ng_v, dim=1)   # each (B, Ng_v)
		fv = uv.mul(dxvv)
		fv = torch.addcmul(fv, vv, dyvv)                  # fv = uv*dxvv + vv*dyvv, (B, Ng_v)

		# --- single projection back to (B, p) ---
		concat = torch.cat([fu, fv], dim=1)               # (B, Ng_u+Ng_v)
		rhs = torch.nn.functional.linear(concat, self.P_proj)  # (B, p)

		return -rhs

	# @torch.jit.export
	def rom_integration(self, output_prev: torch.Tensor, iterations: int) -> torch.Tensor:
		# Step 1: implicit Euler
		start = time.time()
		nonlin_prev = self.rhs(output_prev)            # (B, p)
		print("Time for nonlin prev is " + str(time.time() - start))
		rhs = output_prev + self.dt * nonlin_prev      # (B, p)
		output_new = torch.linalg.lu_solve(self.LU_1st, self.pivots_1st, rhs.transpose(0, 1)).transpose(0, 1)

		stop = 0.0

		# Steps 2..N: BDF2-like
		for _ in range(iterations):
			start = time.time()
			nonlin_new = self.rhs(output_new)
			stop += time.time() - start
			rhs = 4 * output_new - output_prev + 4 * self.dt * nonlin_new - 2 * self.dt * nonlin_prev
			output_prev = output_new
			nonlin_prev = nonlin_new
			output_new = torch.linalg.lu_solve(self.LU_2nd, self.pivots_2nd, rhs.transpose(0, 1)).transpose(0, 1)

		print("Average time for nonlin new is " + str(stop / (iterations)))
		return output_new

	def forward(self, t, state):
		state_rom = state[:, :self.tronc]
		print(f"state_rom size is {state_rom.size()}")

		# Call the JIT-compiled ROM integration
		output_new = self.rom_integration(state_rom, 49)
		rom_output = (output_new - state_rom) / self.dt_skip  # ROM-derived corrections

		return rom_output

class Forecaster(nn.Module):
	def __init__(self, is_augmented, is_phy, N_tensor, redamat, device, dx = 129, dt = 0.04, method='euler', options=None, size = 10, tronc =10):

		super().__init__()

		self.derivative_estimator = DerivativeEstimator(is_augmented=is_augmented, is_phy=is_phy, device = device, redamat = redamat, size = size, tronc = tronc, N_tensor =  N_tensor, dt = dt)
		
		self.method = method
		self.options = options

		self.options = {}
		self.options.update({'method': 'euler'})
		self.options.update({'h': 0.0625})
		self.options.update({'safety': None})
		self.options.update({'regenerate_graph':False})
		
	def forward(self, y0, t):

		self.options.update({'t0': t[0]})
		self.options.update({'t1': t[-1]+0.01})
		self.options.update({'t_eval':t})

		res = odesolve(self.derivative_estimator, y0, options=self.options)
		dims = [1, 2, 0]
		return res.permute(*dims)

def add_vertical_bars(ax, data, lyap_time_interval, chunk_length, color1="black", color2="green"):

	# Lyapunov time bars
	for t in range(lyap_time_interval-1, data.shape[0], lyap_time_interval):
		ax.axvline(x=t, color=color1, linestyle='-', alpha=1, ymin=0, ymax=1)

	# # Chunk length indicator (only after self.chunk)
	# if chunk_length < data.shape[0]:  # Ensure it's within range
	#     ax.axvline(x=chunk_length, color=color2, linestyle='-', linewidth=2, alpha=1, ymin=0, ymax=1)


from pytorch_lightning.loggers import WandbLogger

wandb.login()

class burgers_aphy(pl.LightningModule):

	def __init__(self, is_aug = False, is_phy = True, tronc = 3, size = 27, lr = 1e-3, chunk = 10, batch = 1024, skips = 1):

		super().__init__()

		self.save_hyperparameters()

		self.lr = lr

		print(lr)

		self.best_loss = 1e12

		self.is_phy = is_phy

		self.device1 = "cuda:2"

		print(self.device)

		self.is_aug = is_aug

		self.tronc = tronc

		self.size = size 

		FEM_data = torch.load("pod_freefem_pinball_30.pt")

		podvecs = FEM_data["modes"][:,:100]

		M = FEM_data["M"]

		N = FEM_data["N"][:tronc,:tronc,:tronc]

		# R = FEM_data["R"][:tronc]

		# redamat = torch.linalg.inv(torch.eye(tronc) - FEM_data["dt"]*FEM_data["redmat"][:tronc,:tronc])

		redamat = FEM_data["redmat"][:tronc,:tronc]
		dragvec = FEM_data["dragvec"]
		liftvec = FEM_data["liftvec"]

		pod_tmass = torch.sparse.mm(podvecs.t(),M)

		self.podvecs =torch.tensor(podvecs, device=self.device1).float()

		self.N =torch.tensor(N, device=self.device1).float()
		self.pod_tmass =torch.tensor(pod_tmass, device=self.device1).float()
		self.redamat =torch.tensor(redamat, device=self.device1).float()
		# self.R = torch.tensor(R, device=self.device1).float()

		self.dragvec = torch.tensor(dragvec.squeeze(dim = 0), device=self.device1).float()
		self.liftvec = torch.tensor(liftvec.squeeze(dim = 0), device=self.device1).float()

		self.net = Forecaster( is_augmented = self.is_aug, is_phy = self.is_phy, tronc = self.tronc, size = self.size, device = self.device1, dt = FEM_data["dt"], N_tensor =self.N, redamat = self.redamat)

		self.automatic_optimization = False

		print("finished init")

		self.chunks = chunk
		self.batch = batch 
		self.skips = skips 
		self.current_loss = 1e6

	def training_step(self, batch, batch_idx):

		pass

	def validation_step(self, batch, batch_idx):


		x = batch["states"]
		t =  batch["t"][0]

		print("In validation step")
		print(f"Size of input is {x.size()}")

		nb_ex = batch["states"].size(0)
		nb_steps = batch["states"].size(2)

		# print(f"batch number {batch_idx} contains {nb_ex} examples of size {nb_steps}")

		size1,size2,size3 = x.size()

		retained_modes = x[:,:self.tronc,:]

		start = time.time()

		x_hat = self.net(retained_modes[:,:,0],t)

		print("Time for prediction is " + str(time.time() - start))

		loss_full_POD = nn.functional.mse_loss(x_hat, x[:,:self.tronc,:])
		self.log("val_loss_full_POD", loss_full_POD, on_step=False, on_epoch=True, prog_bar=False, logger=True)
		print("||Qa* -Qa||L2 = " +str(loss_full_POD))

		fig = plt.figure(figsize=(25, 10))
		gs = GridSpec(10, 4, width_ratios=[1, 1, 1, 0.5])

		# Compute drag evolution
		true_drag = torch.einsum('i,mij->mj', self.dragvec, x)
		pred_drag = torch.einsum('i,mij->mj', self.dragvec[:self.tronc], x_hat)


		# If you have lift_rom_vector, do the same for lift
		true_lift = torch.einsum('i,mij->mj', self.liftvec, x)
		pred_lift = torch.einsum('i,mij->mj', self.liftvec[:self.tronc], x_hat)

		for i in range(10):
			ax = fig.add_subplot(gs[i, 0])
			add_vertical_bars(ax, x[0,i, :].cpu().detach().numpy(), 146, self.chunks)
			ax.plot(x[0,i, :].cpu().detach().numpy(), color='red')
			ax.plot(x_hat[0,i, :].cpu().detach().numpy(), color='blue', linestyle='--')
			ax.set_xticks([])
			ax.set_yticks([])

		ax = fig.add_subplot(gs[:5, 1])
		add_vertical_bars(ax, x[0,i, :].cpu().detach().numpy(), 146, self.chunks)
		ax.plot(true_drag[0, :].cpu().detach().numpy(), color='red')
		ax.plot(pred_drag[0, :].cpu().detach().numpy(), color='blue', linestyle='--')
		ax.set_xticks([])
		ax.set_yticks([])

		ax = fig.add_subplot(gs[5:, 1])
		add_vertical_bars(ax, x[0,i, :].cpu().detach().numpy(), 146, self.chunks)
		ax.plot(true_lift[0, :].cpu().detach().numpy(), color='red')
		ax.plot(pred_lift[0, :].cpu().detach().numpy(), color='blue', linestyle='--')
		ax.set_xticks([])
		ax.set_yticks([])

		norm_error = torch.linalg.vector_norm( (x[:,:self.tronc,:] - x_hat), ord = 2, dim = 1 )

		norm_error = norm_error.mean(dim=0)

		initial_error = norm_error[0]

		linx = torch.linspace(0,100,101, device = norm_error.device)
		lyap_exp = 0.00685  # Your computed Lyapunov exponent
		reference_line = initial_error * torch.exp(lyap_exp * linx)

		ax = fig.add_subplot(gs[:, 2])

		add_vertical_bars(ax, norm_error.cpu().detach().numpy(), 146, self.chunks)
		ax.semilogy([0,1000], [0.1, 0.1], color = "black")
		ax.semilogy([0,1000], [1, 1], color = "black")
		ax.semilogy(norm_error.cpu().detach().numpy(), color = "blue", label = "HM divergence from FOM")
		ax.semilogy(reference_line.cpu().numpy(), color = "red", label = "FOM Lyaponov exponent")

		device = x.device

		# Hann window with shape [1, ntimestep] for broadcasting
		hann_window = torch.hann_window(x[:,:self.tronc,:].size(-1), device=device).unsqueeze(0).unsqueeze(0)  # Shape: [1, ntimestep]

		# Apply window to all batches, then compute rFFT along the time dimension (-1)
		fft_1d_ground_truth = torch.abs(torch.fft.rfft( x[:,:self.tronc,:] * hann_window, dim=-1))  # Shape: [batch, dofs, frequencies]
		fft_1d_pred = torch.abs(torch.fft.rfft(x_hat * hann_window, dim=-1))  # Shape: [batch, dofs, frequencies]

		# Average over batch dimension to get [dofs, frequencies]
		fft_1d_ground_truth = fft_1d_ground_truth.mean(dim=0)  # Shape: [dofs, frequencies]
		fft_1d_pred = fft_1d_pred.mean(dim=0)  # Shape: [dofs, frequencies]

		# Convert to NumPy and transpose
		fft_1d_ground_truth = fft_1d_ground_truth.cpu().numpy().T
		fft_1d_pred = fft_1d_pred.cpu().detach().numpy().T

		ax = fig.add_subplot(gs[:5, 3])
		ax.imshow(np.log1p(fft_1d_ground_truth), aspect='auto', interpolation='nearest', cmap='inferno')
		ax.set_xticks([])
		ax.set_yticks([])

		ax = fig.add_subplot(gs[5:, 3])
		ax.imshow(np.log1p(fft_1d_pred), aspect='auto', interpolation='nearest', cmap='inferno')
		ax.set_xticks([])
		ax.set_yticks([])

		plt.tight_layout()
		plt.savefig('pinball_hr_30_pt.png', dpi=100)

		plt.close() 
		plt.clf()

	def test_step(self, batch, batch_idx):
		pass

	def configure_optimizers(self): 
		pass

# Wandb setup
wandb.login()

# Sweep configuration (if using Wandb sweeps)
sweep_config = {
	"name": "my-sweep",
	"method": "grid",
	"metric": {"name": "val_loss_physical", "goal": "minimize"},
	"parameters": {
		"batch_size": {"values": [1024]},
		"model_phy": {"values": [True]},
		"learning_rate": {"values": [5e-3]},
		"tronc": {"values": [20]},
		"size": {"values": [30]},
	},
}

sweep_id = wandb.sweep(sweep_config, project="pinball")

def train(config=None):
	with wandb.init(config=config, project="pinball") as run:
		config = run.config

		wandb_logger = WandbLogger(project="pinball")

		# Load datasets
		loaded_train_dataset = torch.load("dataset_freefem_pinball_30_train.pt")
		loaded_val_dataset = torch.load("dataset_freefem_pinball_30_test.pt")

		# Initialize datasets
		chunk_size = 10 # Adjust as needed
		train_dataset = ChunkedDataset(loaded_train_dataset, chunk_size=chunk_size, skips = 1)
		val_dataset = ChunkedvalDataset(loaded_val_dataset, chunk_size=1000, skips = 100)

		# Test dataset
		sample = train_dataset[0]
		print("Sample 'states' size:", sample["states"].size())
		print("Sample 't' size:", sample["t"].size())

		print("!!!!!!!!!!!!!! dl init  !!!!!!!!!!!!!!")

		# Initialize DataModule
		dl = CustomDataModule(train_dataset, val_dataset, batch_size=config["batch_size"], num_workers=0)

		# Initialize model
		test_model = burgers_aphy(
			is_phy=config["model_phy"],
			is_aug=True,
			size=config["size"],
			lr=config["learning_rate"],
			tronc=config["tronc"]
		)
		test_model.str_name = "freefem_cyl" + str(config["tronc"]) + "_" + str(config["size"])

		# Initialize the callback
		dynamic_data_callback = DynamicDataCallback(
			dataset=train_dataset,
			data_module=dl,
		)

		# Initialize trainer
		trainer = pl.Trainer(
			reload_dataloaders_every_n_epochs =1,
			accelerator="gpu",
			devices=[2],
			logger=wandb_logger,
			log_every_n_steps=1,
			check_val_every_n_epoch=1,
			max_epochs=1,
			enable_checkpointing=False,
			callbacks=[TQDMProgressBar(refresh_rate=1), dynamic_data_callback],
		)

		print("!!!!!!!!!!!!!! trainer init  !!!!!!!!!!!!!!")

		# Start training
		trainer.fit(test_model, dl)

		# Start testing
		trainer.test(test_model, dl)

		print("!!!!!!!!!!!!!! trainer fitted  !!!!!!!!!!!!!!")

		# Clean up
		del dl
		del test_model
		del trainer

		print("!!!!!!!!!!!!!! run is done !!!!!!!!!!!!!!")

		wandb.finish()

# Run the sweep agent
count = 100  # number of runs to execute
wandb.agent(sweep_id, function=train, count=count, project="pinball")