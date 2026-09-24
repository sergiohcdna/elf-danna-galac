
# ###############################################################################
# # COMBINED ANALYSIS (ALL SIMULATIONS/RUNS)                                    #
# #   This scripts replicates the same counts to photon rate and spectra        #
# #   calculations from /notebooks/get_dnde.ipynb, enabling them for the        # 
# #   combination of multiple simulation runs per cluster                       #
# #-----------------------------------------------------------------------------#
# #  Inputs:                                                                    #
# #   1. CSV file with sample's info columns [NAME,REDSHIFT,M200,R200,D_L]      #
# #   2. PATH with all runs containing .FITS.GZ files from dmspectra outputs    #
# #-----------------------------------------------------------------------------#
# #  Outputs:                                                                   #  
# #   - Counts to photon rate (and diff. photon rate dN/dE/dt)                  #
# #   - Spectral Energy distribution (νLν = E² dN/dE/dt)                        #  
# #   - Energy Flux [at Earth] (νFν = E² dN/dE/dt/dS )                          # 
# #   - Total Luminosity  (L = ∫ E (dN/dE/dt) dE ) [at R200]                    #   
# #-----------------------------------------------------------------------------#
# #  Important:                                                                 #
# #   - Install tqdm package if missing (pip install tqdm)                      #
# #   - CSV file cluster names should not include spaces                        #
# #-----------------------------------------------------------------------------#
# #  Use (command-line):                                                        #  
# #   python elfdannagalac/scripts/dnde_luminosity.py --runs-dir /path/to/runs  #  
# #   --runs Run001 Run002 Run003  --clusters-file "/path/to/clusters.csv"      #
# #   --name-col NAME --m200-col M200  --redshift-col Z  --distance-col D_L     #
# #   --max-clusters 15  --n-sim 200000  --channel tau                          #
# #   --outdir "/path/results/tau/"  --file-pattern "{cluster}decay.fits.gz"    # 
# #-----------------------------------------------------------------------------#
# #                      THE ELF-DANNA-GALAC Task force                         #
# #                      - Arlette Melo Galindo                                 #
# #                      - Miguel A. Sánchez Conde                              #
# #                      - Sergio Hernández Cadena                              #
# #-----------------------------------------------------------------------------#
# #                                                          June-2026          #
# ###############################################################################



# import sys
# import os
# from pathlib import Path
# import numpy as np
# import astropy.units as u
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.cosmology import Planck18 as cosmo
# from ebltable.tau_from_model import OptDepth
# import pandas as pd
# from tqdm import tqdm  #barras de progreso
# from elfdannagalac.dmspectrum import dmspectrum, dmdecayNorm, Qe_dmdecay, Qe_dmanna, weigths_PL_sim

# import argparse
# import warnings

# # warnings.filterwarnings('ignore')




# class ClusterAnalysis:

# # ============================================================
# # Initialization function: Parameters received from the user 
# #                          and saved for future processing.
# # ============================================================
#     def __init__(self, 
#                  cluster_name: str, 
#                  runs_dir: str, 
#                  runs_list: list,
#                  file_pattern: str = "{cluster}.fits.gz",
#                  dm_mass: u.Quantity = 1e4*u.GeV, 
#                  process: str = 'decay',
#                  channel: str = 'tau', 
#                  tau_dm: u.Quantity = 1e27*u.s,  # DM lifetime
#                  m200: u.Quantity = None, 
#                  redshift: float = None, 
#                  distance_luminosity: u.Quantity = None,
#                  n_sim_fixed: int = None):
        
#         self.cluster_name = cluster_name
#         self.dm_mass = dm_mass
#         self.process = process
#         self.channel = channel
#         self.tau_dm = tau_dm
#         self.m200 = m200
#         self.redshift = redshift
#         self.distance_luminosity = distance_luminosity
#         self.file_pattern = file_pattern
#         self.runs_dir = Path(runs_dir).resolve()
#         self.n_sim_fixed = n_sim_fixed
    
#         self._load_all_runs(runs_list)

# # ====================================================================
# # Data loading function: Loading of multiple runs (for all clusters).
# # ====================================================================     
#     def _load_all_runs(self, runs_list: list):
#         self.e_gamma_list = []
#         self.e_weights_list = []
#         self.e_source_list = []
#         self.total_particles = 0
        
#         # Searching for files in each run from runs directory
#         for run in runs_list:
#             run_path = self.runs_dir / run
#             filename = self.file_pattern.format(cluster=self.cluster_name)
#             filepath = run_path / filename
            
#             # Seraching for files if specified name pattern 
#             # does not match
#             if not filepath.exists():
#                 fits_files = list(run_path.glob(f"*{self.cluster_name}*.fits.gz"))
#                 if fits_files:
#                     filepath = fits_files[0]
#                 else:
#                     fits_files = list(run_path.glob(f"*{self.cluster_name}*.fits"))
#                     if fits_files:
#                         filepath = fits_files[0]
#                     else:
#                         continue

#         # Reading .fits files
#             try:
#                 hdu = fits.open(filepath)[1]
#                 eng_unit = u.Unit(hdu.data.columns["E"].unit)
#                 e_gamma = hdu.data["E"] * eng_unit
#                 e_weights = hdu.data["W"]
#                 e_source = hdu.data["E0"] * eng_unit
                
#                 self.e_gamma_list.append(e_gamma)
#                 self.e_weights_list.append(e_weights)
#                 self.e_source_list.append(e_source)
#                 self.total_particles += len(e_gamma)
#             except Exception as e:
#                 print(f"  Error loading {filepath}: {e}")
#                 continue
        
#         if len(self.e_gamma_list) == 0:
#             raise ValueError(f"No files found for {self.cluster_name}")
        
#         # Combination of all runs data in a single array
#         self.e_gamma = np.concatenate(self.e_gamma_list)
#         self.e_weights = np.concatenate(self.e_weights_list)
#         self.e_source = np.concatenate(self.e_source_list)
        
#         # print(f"  Total particles: {self.total_particles}")

#     # ============================================================        
#     # DM Spectra configuration: DM mass and decay
#     # (or anna) channels can be passed as an argument 
#     # ============================================================      
#     def run_analysis(self, emin_sim=100*u.GeV, emax_sim=100*u.TeV,
#                      emin_dm=100*u.GeV, emax_dm=100*u.TeV):
#         # ===============
#         # DM Spectra
#         # ===============
#         self.cosmix_spec = dmspectrum(
#             self.dm_mass.value,
#             emin_dm.value,
#             emax_dm.value,
#             self.channel,
#             process=self.process
#         )
        
#         self.dnde_e_real = self.cosmix_spec.spectrum()
#         self.dm_interp = self.cosmix_spec.interpolator()

#         # =============================
#         # DM Spectra Normalization
#         # =============================
#         self.dmNorm = dmdecayNorm(
#             self.dm_interp,
#             self.cosmix_spec.mass * u.GeV,
#             emin_dm,
#             emax_dm,
#         )
#         print(f"  dmNorm: {self.dmNorm:.6e}")
        
#         # ===============================================
#         # Injection rate of electrons from DM anna/decay
#         # ===============================================
#         if self.m200 is not None:
#             if self.process == 'decay':
#                 self.ndot_exp = Qe_dmdecay(
#                     self.dm_mass,
#                     self.m200,
#                     self.tau_dm,
#                     self.dmNorm
#                 )
#             else:  # 'anna'
#                 self.ndot_exp = Qe_dmanna(
#                     self.dm_mass,
#                     self.m200,
#                     self.tau_dm,
#                     self.dmNorm
#                 )
#         else:
#             # Default value (A193 decay)
#             print(f"  !!! WARNING: No M200 value found for {self.cluster_name}")
#             print(f"     Using Abell 193 default value (2.40435e14 Msun)")
#             self.ndot_exp = 4.118547026472017e+38 / u.s
        
#         print(f"  Process: {self.process}")
#         print(f"  dmNorm: {self.dmNorm:.6e}")
#         print(f"  ndot_exp: {self.ndot_exp:.6e}") 

#         # ============================================================
#         # Weights: simulated and expected spectrum weights + CRPropa weights
#         # ============================================================
#         self.emin_sim = emin_sim
#         self.emax_sim = emax_sim
         
#         if self.n_sim_fixed is not None:
#             n_sim = self.n_sim_fixed
#             print(f"  Using n_sim: {n_sim}")
#         else:
#             n_sim = self.total_particles
#             print(f"  Using n_sim = total_particles: {n_sim}")
        
#         self.dm_weights = weigths_PL_sim(
#             n_sim,
#             self.ndot_exp,
#             self.dm_mass,
#             self.e_source,
#             emin_sim,
#             emax_sim,
#             self.dm_interp,
#             self.dmNorm,
#         )
        
#         self.total_weights = self.e_weights * self.dm_weights
#         print(f"  dm_weights: min={np.min(self.dm_weights):.6e}, max={np.max(self.dm_weights):.6e}, mean={np.mean(self.dm_weights):.6e}")
#         print(f"  total_weights: min={np.min(self.total_weights):.6e}, max={np.max(self.total_weights):.6e}, mean={np.mean(self.total_weights):.6e}")
        
#         # ============================
#         # Photon rate calculation
#         # ============================
#         self.compute_spectra()
        
#     # ============================================================
#     # Spectra Calculation: Binning and Histogram creation
#     # ============================================================   
#     def compute_spectra(self, delta_loge=0.2):
        
        
#         log_emin = -6.0
#         log_emax = 5.0 + delta_loge
#         self.energy_bins = 10 ** np.arange(log_emin, log_emax, delta_loge) * u.GeV
#         self.bin_widths = np.diff(self.energy_bins)
#         self.bin_centers = (self.energy_bins[:-1] + self.energy_bins[1:]) / 2
#         print(f"  Energy bins: {len(self.energy_bins)-1} bins from {self.energy_bins[0]:.2e} to {self.energy_bins[-1]:.2e}")
        
#         # ===============
#         # Counts rate
#         # ===============
#         counts_hist, _ = np.histogram(
#             self.e_gamma.value,
#             bins=self.energy_bins.value,
#             weights=self.total_weights.value
#         )
#         self.counts_hist = counts_hist
#         self.counts_rate = counts_hist / u.s  # counts/s
        
#         # Differential photon rate
#         # dN/dE/dt = counts_rate / bin_widths
#         self.dnde_dt = self.counts_rate / self.bin_widths
        
#         # νLν (E²dN/dE/dt)
#         self.e2_dnde_dt = self.dnde_dt * self.bin_centers**2
        
#         # ============================================================
#         # νFν (E²dN/dE/dt) [at Earth]
#         # including EBL attenuation
#         # ============================================================
        
#         if self.redshift is not None and self.distance_luminosity is not None:
#             ebl_model = OptDepth.readmodel(model='saldana-lopez')
#             atten = np.exp(-1. * ebl_model.opt_depth(
#                 self.redshift, 
#                 self.e_gamma.to(u.TeV).value
#             ))
            
#             # Total weight for NuFnu: e_weights * dmw * e_gamma^2 * atten
#             total_weight_nu = self.total_weights * self.e_gamma**2 * atten
            
#             counts_hist_nu, _ = np.histogram(
#                 self.e_gamma.value,
#                 bins=self.energy_bins.value,
#                 weights=total_weight_nu.value
#             )
            
#             # Surface 
#             surface = 4 * np.pi * (self.distance_luminosity.to(u.cm))**2
            
#             # νFν = counts_hist*(1+z)*rate_unit/binwidths/surface
#             self.nuFnu = counts_hist_nu * (1 + self.redshift) / (self.bin_widths * surface) * (1/u.s)
            
#             print(f"  nuFnu [1/(GeV s cm2)] calculated succesfully")
#             idx_100 = np.argmin(np.abs(self.bin_centers.value - 100))
#             print(f"  nuFnu (100 GeV): {self.nuFnu[idx_100]:.6e}")
#         else:
#             self.nuFnu = None
#             print(f"  nuFnu: redshift/luminosity distance missing")

#     # ============================================================
#     # Luminosity Calculation: Calculates total luminosity
#     #                         integrating E * dN/dE/dt spectrum.
#     # ============================================================
#     def compute_luminosity(self, emin=1e-6*u.GeV, emax=1e5*u.GeV):
    
#         # Selecting bins within integration range
#         mask = (self.bin_centers >= emin) & (self.bin_centers <= emax)
        
#         if not np.any(mask):
#             print(f" !!! No bins in range {emin} - {emax} for {self.cluster_name}")
#             self.luminosity = 0 * u.GeV / u.s
#             self.luminosity_erg = 0 * u.erg / u.s
#             return self.luminosity
        
#         # E * dN/dE/dt 
#         e_dnde = self.dnde_dt * self.bin_centers
#         bin_widths = self.bin_widths[mask]
        
#         # Integration
#         self.luminosity = np.sum(e_dnde[mask] * bin_widths)
#         self.luminosity_erg = self.luminosity.to(u.erg / u.s)
        
#         print(f"  Luminosity ({self.cluster_name}): {self.luminosity:.6e}")
        
#         return self.luminosity

#     # ============================================================
#     # SAVING RESULTS: Plots and spectra results are saved in 
#     #                 the path indicated by the user.
#     # ============================================================
#     def save_results(self, output_dir: str):

#         output_dir = Path(output_dir) / self.cluster_name
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         results_df = pd.DataFrame({
#             'E_center_GeV': self.bin_centers.value,
#             'E_low_GeV': self.energy_bins[:-1].value,
#             'E_high_GeV': self.energy_bins[1:].value,
#             'counts_hist': self.counts_hist,
#             'counts_rate_1per_s': self.counts_rate.value,
#             'dNdEdt_GeV-1_s-1': self.dnde_dt.value,
#             'E2_dNdEdt_GeV_s-1': self.e2_dnde_dt.value,
#         })
        
        
#         if self.nuFnu is not None:
#             results_df['nuFnu_1_per_GeV_s_cm2'] = self.nuFnu.value
            
#         results_df.to_csv(output_dir / 'spectra.csv', index=False)
        
#         if not hasattr(self, 'luminosity'):
#             self.compute_luminosity()

#         metadata = {
#             'cluster_name': self.cluster_name,
#             'dm_mass_GeV': self.dm_mass.value,
#             'process': self.process,
#             'channel': self.channel,
#             'tau_dm_s': self.tau_dm.value,
#             'm200_Msun': self.m200.value if self.m200 is not None else None,
#             'redshift': self.redshift,
#             'distance_luminosity_Mpc': self.distance_luminosity.value if self.distance_luminosity is not None else None,
#             'total_particles': self.total_particles,
#             'n_sim_used': self.n_sim_fixed if self.n_sim_fixed is not None else self.total_particles,
#             'ndot_exp_1_per_s': self.ndot_exp.value if self.ndot_exp is not None else None,
#             'dmNorm': self.dmNorm,
#             'n_bins': len(self.energy_bins) - 1,
#             'delta_loge': np.log10(self.energy_bins[1].value) - np.log10(self.energy_bins[0].value),
#             'luminosity_GeV_s': self.luminosity.value,
#             'luminosity_erg_s': self.luminosity_erg.value,
#         }
#         pd.Series(metadata).to_csv(output_dir / 'metadata.csv')

#     # ============================================================
#     # Generating the PLOTS
#     # ============================================================
#     def plot_spectra(self, output_dir: str):
#         output_dir = Path(output_dir) / self.cluster_name
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Counts rate
#         fig1, ax1 = plt.subplots(figsize=(9, 8))
#         ax1.stairs(self.counts_rate.value, self.energy_bins.value,
#                    facecolor="pink", alpha=0.65)
#         ax1.set_xscale("log")
#         ax1.set_yscale("log")
#         ax1.set_xlabel("Energy [GeV]", fontsize=20)
#         ax1.set_ylabel("Counts rate [1/s]", fontsize=20)
#         fig1.savefig(output_dir / 'counts_rate.png', dpi=150, bbox_inches='tight')
#         plt.close(fig1)
        
#         # dN/dE/dt
#         fig2, ax2 = plt.subplots(figsize=(9, 8))
#         ax2.stairs(self.dnde_dt.value, self.energy_bins.value,
#                    facecolor="gray", fill=True, alpha=0.35)
#         ax2.set_xscale("log")
#         ax2.set_yscale("log")
#         ax2.set_xlabel("Energy [GeV]", fontsize=20)
#         ax2.set_ylabel(r"$\frac{{\rm d}N}{{\rm d}E {\rm d}t}~[{\rm GeV}^{-1}~{\rm s}^{-1}]$", fontsize=20)
#         fig2.savefig(output_dir / 'dNdEdt.png', dpi=150, bbox_inches='tight')
#         plt.close(fig2)
        
#         # E² dN/dE/dt
#         fig3, ax3 = plt.subplots(figsize=(9, 8))
#         ax3.stairs(self.e2_dnde_dt.value, self.energy_bins.value,
#                    facecolor="gray", fill=True, alpha=0.35)
#         ax3.set_xscale("log")
#         ax3.set_yscale("log")
#         ax3.set_xlabel("Energy [GeV]", fontsize=20)
#         ax3.set_ylabel(r"$E^2\frac{{\rm d}N}{{\rm d}E {\rm d}t}~[{\rm GeV}~{\rm s}^{-1}]$", fontsize=20)
#         fig3.savefig(output_dir / 'E2dNdEdt.png', dpi=150, bbox_inches='tight')
#         plt.close(fig3)
        
#         # νFν [ GeV s⁻¹ cm⁻²]
#         if self.nuFnu is not None:
#             fig4, ax4 = plt.subplots(figsize=(9, 8))
#             ax4.stairs(self.nuFnu.value, 
#                        self.energy_bins.value,
#                        facecolor="pink", fill=True, alpha=0.65,
#                        label=self.cluster_name)
#             ax4.axvspan(xmin=0.02, xmax=300, color='gray', alpha=0.3, label="Fermi-LAT")
#             ax4.axvspan(xmin=300, xmax=1e5, color='blue', alpha=0.3, label="LHAASO")
#             ax4.set_xscale("log")
#             ax4.set_yscale("log")
#             ax4.set_xlabel("Energy [GeV]", fontsize=20)
#             ax4.set_ylabel(r"$\nu F_\nu~[{\rm GeV}~{\rm s}^{-1}~{\rm cm}^{-2}]$", fontsize=20)
#             ax4.set_title(f"{self.cluster_name} - DM {self.process} {self.channel} channel", fontsize=16)
#             ax4.legend(loc="best", fontsize=14)
#             fig4.savefig(output_dir / 'nuFnu.png', dpi=150, bbox_inches='tight')
#             plt.close(fig4)

# # ============================================================
# # Loading CLUSTER info from .csv file
# # ============================================================
# def load_clusters_info(clusters_file, name_column='NAME',
#                        m200_column='M200',redshift_column='REDSHIFT',
#                        distance_column='D_L'):
    
#     clusters_file = Path(clusters_file).resolve()
#     if not clusters_file.exists():
#         raise FileNotFoundError(f"!!! File not found: {clusters_file}")
    
#     df = pd.read_csv(clusters_file)
    
#     # WARNINGS
#     if name_column not in df.columns:
#         raise ValueError(f" !!! Column '{name_column}' not found in {clusters_file}")
    
#     if m200_column not in df.columns:
#         print(f" !!! Column '{m200_column}' not found in {clusters_file}")
#         print(f"  Default value:  M200 = 2.40435e14 Msun (for all clusters)")
#         df[m200_column] = 2.40435e14
    
#     if distance_column not in df.columns and redshift_column in df.columns:
#         df[distance_column] = cosmo.luminosity_distance(df[redshift_column].values).value
    
#     return df

# # ============================================================
# #  Command line arguments
# # ============================================================
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Luminosity and Spectrum calculation from Photon counts")
#     parser.add_argument('--runs-dir', type=str, required=True, help="Simulation runs directory path")
#     parser.add_argument('--runs', nargs='+', required=True, help="Names of each runs folders")
#     parser.add_argument('--clusters-file', type=str, required=True, help="CSV file with clusters sample info")
#     parser.add_argument('--outdir', type=str, default="./dm_analysis_results", help="Output savig directory")
#     parser.add_argument('--name-col', type=str, default="NAME", help="Name column label in CSV file")
#     parser.add_argument('--m200-col', type=str, default="M200", help="M200 column label in CSV file")
#     parser.add_argument('--redshift-col', type=str, default="REDSHIFT", help="Redshift column label in CSV file")
#     parser.add_argument('--distance-col', type=str, default="D_L", help="Distance column label in CSV file")
#     parser.add_argument('--file-pattern', type=str, default="{cluster}.fits.gz", help="dmspcetra simulation output file pattern for each cluster")
#     parser.add_argument('--dm-mass', type=float, default=10000, help="DM mass in GeV")
#     parser.add_argument('--channel', type=str, default='tau', help="Annihilation/decay channel")
#     parser.add_argument('--tau-dm', type=float, default=1e27, help="DM particle lifetime") 
#     parser.add_argument('--emin-dm', type=float, default=100)
#     parser.add_argument('--emax-dm', type=float, default=100000)
#     parser.add_argument('--emin-sim', type=float, default=100)
#     parser.add_argument('--emax-sim', type=float, default=100000)
#     parser.add_argument('--max-clusters', type=int, default=None, help="Max number of clusters (per run) to analyze")
#     parser.add_argument('--no-plots', action='store_true')
#     parser.add_argument('--delta-loge', type=float, default=0.2, help="Logarithmic spacing of bins") 
#     parser.add_argument('--n-sim', type=int, default=200000, help="Number of simulated particles per run")
#     return parser.parse_args()

# # ============================================================
# # main function
# # ============================================================
# def main():
#     args = parse_arguments()
    
#     dm_mass = args.dm_mass * u.GeV
#     tau_dm = args.tau_dm * u.s
#     emin_dm = args.emin_dm * u.GeV
#     emax_dm = args.emax_dm * u.GeV
#     emin_sim = args.emin_sim * u.GeV
#     emax_sim = args.emax_sim * u.GeV
    
#     print("=" * 70)
#     print("CLUSTER DM ANALYSIS ")
#     print("=" * 70)
    
#     clusters_df = load_clusters_info(
#         args.clusters_file,
#         name_column=args.name_col,
#         m200_column=args.m200_col,
#         redshift_column=args.redshift_col,
#         distance_column=args.distance_col
#     )
#     print(f"  Loaded clusters: {len(clusters_df)}")
    
#     if args.max_clusters:
#         clusters_df = clusters_df.head(args.max_clusters)
    
#     output_dir = Path(args.outdir).resolve()
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     results = []
#     luminosity_list = []

#     for idx, row in tqdm(clusters_df.iterrows(), total=len(clusters_df), desc="Clusters"):
#         cluster_name = row[args.name_col]
        
#         try:
#             analysis = ClusterAnalysis(
#                 cluster_name=cluster_name,
#                 runs_dir=args.runs_dir,
#                 runs_list=args.runs,
#                 file_pattern=args.file_pattern,
#                 dm_mass=dm_mass,
#                 channel=args.channel,
#                 tau_dm=tau_dm,
#                 m200=row.get(args.m200_col, None) * u.Msun if args.m200_col in row else None,
#                 redshift=row.get(args.redshift_col, None),
#                 distance_luminosity=row.get(args.distance_col, None) * u.Mpc if args.distance_col in row else None,
#                 n_sim_fixed=args.n_sim
#             )
            
#             analysis.run_analysis(
#                 emin_sim=emin_sim,
#                 emax_sim=emax_sim,
#                 emin_dm=emin_dm,
#                 emax_dm=emax_dm
#             )
            
#             analysis.compute_luminosity()
#             analysis.save_results(output_dir=output_dir)
            
#             if not args.no_plots:
#                 analysis.plot_spectra(output_dir=output_dir)

#             luminosity_list.append({
#                 'cluster': cluster_name,
#                 'm200_Msun': analysis.m200.value if analysis.m200 is not None else None,
#                 'luminosity_GeV_s': analysis.luminosity.value,
#                 'luminosity_erg_s': analysis.luminosity_erg.value,
#             })
            
#             results.append({
#                 'cluster': cluster_name,
#                 'particles': analysis.total_particles,
#                 'n_sim_used': args.n_sim if args.n_sim is not None else analysis.total_particles,
#                 'ndot_exp': analysis.ndot_exp.value if analysis.ndot_exp is not None else None,
#                 'dmNorm': analysis.dmNorm,
#                 'luminosity_GeV_s': analysis.luminosity.value,    
#                 'luminosity_erg_s': analysis.luminosity_erg.value,
#             })
            
#         except Exception as e:
#             print(f"\n  Error processing {cluster_name}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     luminosity_df = pd.DataFrame(luminosity_list)
#     luminosity_df.to_csv(output_dir / 'clusters_luminosities.csv', index=False)
#     print(f"\n Luminosities saved in: {output_dir / 'clusters_luminosities.csv'}")
    
#     summary_df = pd.DataFrame(results)
#     summary_df.to_csv(output_dir / 'analysis_summary.csv', index=False)
    
#     print("\n" + "=" * 70)
#     print("ANALYSIS COMPLETED! :)")
#     print("=" * 70)
#     print(f"Results saved in: {output_dir}")


# if __name__ == "__main__":
#     main()



###############################################################################
# COMBINED ANALYSIS (ALL SIMULATIONS/RUNS)                                    #
#   This scripts replicates the same counts to photon rate and spectra        #
#   calculations from /notebooks/get_dnde.ipynb, enabling them for the        # 
#   combination of multiple simulation runs per cluster                       #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#                                                          June-2026          #
###############################################################################

import sys
import os
from pathlib import Path
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from ebltable.tau_from_model import OptDepth
import pandas as pd
from tqdm import tqdm
from elfdannagalac.dmspectrum import dmspectrum, dmdecayNorm, Qe_dmdecay, Qe_dmanna, weigths_PL_sim
from scipy.interpolate import interp1d

import argparse
import warnings


class ClusterAnalysis:

    def __init__(self, 
                 cluster_name: str, 
                 runs_dir: str, 
                 runs_list: list,
                 file_pattern: str = "{cluster}.fits.gz",
                 dm_mass: u.Quantity = 1e4*u.GeV, 
                 process: str = 'decay',
                 channel: str = 'tau', 
                 tau_dm: u.Quantity = 1e27*u.s,
                 m200: u.Quantity = None, 
                 redshift: float = None, 
                 distance_luminosity: u.Quantity = None,
                 n_sim_fixed: int = None):
        
        self.cluster_name = cluster_name
        self.dm_mass = dm_mass
        self.process = process
        self.channel = channel
        self.tau_dm = tau_dm
        self.m200 = m200
        self.redshift = redshift
        self.distance_luminosity = distance_luminosity
        self.file_pattern = file_pattern
        self.runs_dir = Path(runs_dir).resolve()
        self.n_sim_fixed = n_sim_fixed
    
        self._load_all_runs(runs_list)

    def _load_all_runs(self, runs_list: list):
        self.e_gamma_list = []
        self.e_weights_list = []
        self.e_source_list = []
        self.total_particles = 0
        
        for run in runs_list:
            run_path = self.runs_dir / run
            filename = self.file_pattern.format(cluster=self.cluster_name)
            filepath = run_path / filename
            
            if not filepath.exists():
                fits_files = list(run_path.glob(f"*{self.cluster_name}*.fits.gz"))
                if fits_files:
                    filepath = fits_files[0]
                else:
                    fits_files = list(run_path.glob(f"*{self.cluster_name}*.fits"))
                    if fits_files:
                        filepath = fits_files[0]
                    else:
                        continue

            try:
                hdu = fits.open(filepath)[1]
                df = pd.DataFrame(hdu.data)
                
                # mask = ~df.duplicated(subset='SN0', keep='first')
                # df_filtered = df[mask]
                
                # e_gamma = df_filtered["E"].values
                # e_weights = df_filtered["W"].values
                # e_source = df_filtered["E0"].values
                e_source = df["E0"]
                e_gamma = df["E"].values
                e_weights = df["W"].values

                self.e_gamma_list.append(e_gamma)
                self.e_weights_list.append(e_weights)
                self.e_source_list.append(e_source)
                self.total_particles += len(e_gamma)
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue
        
        if len(self.e_gamma_list) == 0:
            raise ValueError(f"No files found for {self.cluster_name}")
        
        self.e_gamma = np.concatenate(self.e_gamma_list)
        self.e_weights = np.concatenate(self.e_weights_list)
        self.e_source = np.concatenate(self.e_source_list)

    def run_analysis(self):
        
        # PARAMETROS FIJOS PARA ELECTRONES (como en el script de referencia)
        dmmass = self.dm_mass
        emin_dm = 100 * u.GeV
        emax_dm = 0.45 * dmmass
        emin_sim = 100 * u.GeV
        emax_sim = 0.5 * dmmass
        tau_dm = self.tau_dm
        
        # ESPECTRO DE DM PARA ELECTRONES
        self.cosmix_spec_e = dmspectrum(
            dmmass.value,
            emin_dm.value,
            emax_dm.value,
            self.channel,
            process=self.process
        )
        
        self.dm_interp_e = self.cosmix_spec_e.interpolator()

        self.dmNorm = dmdecayNorm(
            self.dm_interp_e,
            dmmass,
            emin_dm,
            emax_dm,
        )
        print(f"  dmNorm: {self.dmNorm:.6e}")
        
        if self.m200 is not None:
            if self.process == 'decay':
                self.ndot_exp = Qe_dmdecay(
                    dmmass,
                    self.m200,
                    tau_dm,
                    self.dmNorm
                )
            else:
                self.ndot_exp = Qe_dmanna(
                    dmmass,
                    self.m200,
                    tau_dm,
                    self.dmNorm
                )
        else:
            print(f"  !!! WARNING: No M200 value found for {self.cluster_name}")
            self.ndot_exp = 4.118547026472017e+38 / u.s
        
        print(f"  ndot_exp: {self.ndot_exp:.6e}") 

        # PESOS PARA ELECTRONES
        if self.n_sim_fixed is not None:
            n_sim = self.n_sim_fixed
        else:
            n_sim = len(self.e_source)
        
        self.dm_weights_e = weigths_PL_sim(
            n_sim,
            self.ndot_exp,
            dmmass,
            self.e_source * u.GeV,
            emin_sim,
            emax_sim,
            self.dm_interp_e,
            self.dmNorm,
        )
        
        # ============================================================
        # emax_sim = np.max(e_gamma_total) * u.GeV
        emin_sim_gamma = 100 * u.GeV                     # : fijo en 100 GeV
        emax_sim_gamma = np.max(self.e_gamma) * u.GeV
        
        self.dm_weights_gamma = weigths_PL_sim(
            n_sim,
            self.ndot_exp,
            dmmass,
            self.e_source * u.GeV,
            emin_sim_gamma,
            emax_sim_gamma,
            self.dm_interp_e,
            self.dmNorm,
        )
        
        self.total_weights = self.e_weights * self.dm_weights_gamma
        
        # ESPECTRO DE FOTONES DE COSMIXS (con limites de los fotones simulados)
        emin_dm_gamma = np.min(self.e_gamma) * u.GeV
        emax_dm_gamma = np.max(self.e_gamma) * u.GeV
        
        self.cosmix_spec_gamma = dmspectrum(
            dmmass.value,
            emin_dm_gamma.value,
            emax_dm_gamma.value,
            self.channel,
            process=self.process,
            particle="gamma"
        )
        
        self.gamma_spectra = self.cosmix_spec_gamma.spectrum()
        self.gamma_engs = self.cosmix_spec_gamma.engs
        
        self.compute_spectra()

    def compute_spectra(self, delta_loge=0.1):
        
        # E² dN/dEdt (Intrinsic) - bins basados en fotones simulados
        log_emin = np.log10(np.min(self.e_gamma))
        log_emax = np.log10(np.max(self.e_gamma)) + delta_loge
        self.energy_bins = 10 ** np.arange(log_emin, log_emax, delta_loge) * u.GeV
        self.bin_widths = np.diff(self.energy_bins)
        self.bin_centers = (self.energy_bins[:-1] + self.energy_bins[1:]) / 2
        print(f"  Energy bins: {len(self.energy_bins)-1} bins from {self.energy_bins[0]:.2e} to {self.energy_bins[-1]:.2e}")
        
        counts_hist, _ = np.histogram(
            self.e_gamma,
            bins=self.energy_bins.value,
            weights=self.total_weights.value
        )
        self.counts_hist = counts_hist
        self.counts_rate = counts_hist / u.s
        self.dnde_dt = self.counts_rate / self.bin_widths
        self.e2_dnde_dt = self.dnde_dt * self.bin_centers**2
        
        # νFν (at Earth) - bins FIJOS -6 a 5
        delta_loge_nu = 0.1
        self.energy_bins_nu = 10 ** np.arange(-6.0, 5.0 + delta_loge_nu, delta_loge_nu) * u.GeV
        self.bin_widths_nu = np.diff(self.energy_bins_nu)
        self.bin_centers_nu = (self.energy_bins_nu[:-1] + self.energy_bins_nu[1:]) / 2
        
        if self.redshift is not None and self.distance_luminosity is not None:
            ebl_model = OptDepth.readmodel(model='saldana-lopez')
            surface = 4 * np.pi * (self.distance_luminosity.to(u.cm))**2
            m200_GeV = self.m200.to(u.GeV, equivalencies=u.mass_energy())
            
            # νFν SIM - SIN unidades
            atten_sim = np.exp(-1. * ebl_model.opt_depth(
                self.redshift, 
                (self.e_gamma * u.GeV).to(u.TeV).value
            ))
            
            total_weight_nu = self.total_weights.value * (self.e_gamma)**2 * atten_sim
            
            counts_hist_nu, _ = np.histogram(
                self.e_gamma,
                bins=self.energy_bins_nu.value,
                weights=total_weight_nu
            )
            
            nuFnu_sim = counts_hist_nu * (1 + self.redshift) / (self.bin_widths_nu.value * surface.value)
            self.nuFnu_sim = nuFnu_sim * (u.GeV / u.s / u.cm**2)
            
            # νFν PROMPT - SIN unidades
            atten_prompt = np.exp(-1. * ebl_model.opt_depth(
                self.redshift, 
                (self.gamma_engs * u.GeV).to(u.TeV).value
            ))
            
            nuFnu_prompt = (self.gamma_spectra * m200_GeV.value / (self.tau_dm.value * self.dm_mass.value)) * (self.gamma_engs)**2 * (1 + self.redshift) / surface.value
            nuFnu_prompt *= atten_prompt
            self.nuFnu_prompt = nuFnu_prompt * (u.GeV / u.s / u.cm**2)
            
            # νFν TOTAL - SUMA DE NUMEROS SIN UNIDADES
            interp_sim = interp1d(
                self.bin_centers_nu.value,
                nuFnu_sim,
                kind='linear',
                bounds_error=False,
                fill_value=0
            )
            nuFnu_sim_interp = interp_sim(self.gamma_engs)
            
            nuFnu_total = nuFnu_prompt + nuFnu_sim_interp
            self.nuFnu_total = nuFnu_total * (u.GeV / u.s / u.cm**2)
            
            print(f"  nuFnu calculated successfully")
        else:
            self.nuFnu_sim = None
            self.nuFnu_prompt = None
            self.nuFnu_total = None
            print(f"  nuFnu: redshift/luminosity distance missing")

    def compute_luminosity(self, emin=1e-6*u.GeV, emax=1e5*u.GeV):
        mask = (self.bin_centers >= emin) & (self.bin_centers <= emax)
        
        if not np.any(mask):
            print(f" !!! No bins in range {emin} - {emax} for {self.cluster_name}")
            self.luminosity = 0 * u.GeV / u.s
            self.luminosity_erg = 0 * u.erg / u.s
            return self.luminosity
        
        e_dnde = self.dnde_dt * self.bin_centers
        bin_widths = self.bin_widths[mask]
        
        self.luminosity = np.sum(e_dnde[mask] * bin_widths)
        self.luminosity_erg = self.luminosity.to(u.erg / u.s)
        
        print(f"  Luminosity ({self.cluster_name}): {self.luminosity:.6e}")
        return self.luminosity

    def save_results(self, output_dir: str):
        output_dir = Path(output_dir) / self.cluster_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'luminosity'):
            self.compute_luminosity()
        
        electrons_df = pd.DataFrame({
            'E_center_GeV': self.bin_centers.value,
            'E_low_GeV': self.energy_bins[:-1].value,
            'E_high_GeV': self.energy_bins[1:].value,
            'counts_hist': self.counts_hist,
            'counts_rate_1per_s': self.counts_rate.value,
            'dNdEdt_GeV-1_s-1': self.dnde_dt.value,
            'E2_dNdEdt_GeV_s-1': self.e2_dnde_dt.value,
        })
        electrons_df.to_csv(output_dir / 'spectra_electrons.csv', index=False)
        
        photons_df = pd.DataFrame({
            'luminosity_GeV_s': [self.luminosity.value],
            'luminosity_erg_s': [self.luminosity_erg.value],
        })
        photons_df.to_csv(output_dir / 'spectra_photons.csv', index=False)
        
        if self.nuFnu_sim is not None:
            interp_prompt = interp1d(
                self.gamma_engs,
                self.nuFnu_prompt.value,
                kind='linear',
                bounds_error=False,
                fill_value=0
            )
            nuFnu_prompt_grid = interp_prompt(self.bin_centers_nu.value)
            
            interp_total = interp1d(
                self.gamma_engs,
                self.nuFnu_total.value,
                kind='linear',
                bounds_error=False,
                fill_value=0
            )
            nuFnu_total_grid = interp_total(self.bin_centers_nu.value)
            
            nu_df = pd.DataFrame({
                'E_center_GeV': self.bin_centers_nu.value,
                'E_low_GeV': self.energy_bins_nu[:-1].value,
                'E_high_GeV': self.energy_bins_nu[1:].value,
                'nuFnu_sim_1_per_GeV_s_cm2': self.nuFnu_sim.value,
                'nuFnu_prompt_1_per_GeV_s_cm2': nuFnu_prompt_grid,
                'nuFnu_total_1_per_GeV_s_cm2': nuFnu_total_grid,
            })
            nu_df.to_csv(output_dir / 'nuFnu.csv', index=False)

        metadata = {
            'cluster_name': self.cluster_name,
            'dm_mass_GeV': self.dm_mass.value,
            'process': self.process,
            'channel': self.channel,
            'tau_dm_s': self.tau_dm.value,
            'm200_Msun': self.m200.value if self.m200 is not None else None,
            'redshift': self.redshift,
            'distance_luminosity_Mpc': self.distance_luminosity.value if self.distance_luminosity is not None else None,
            'total_particles': self.total_particles,
            'n_sim_used': self.n_sim_fixed if self.n_sim_fixed is not None else self.total_particles,
            'ndot_exp_1_per_s': self.ndot_exp.value if self.ndot_exp is not None else None,
            'dmNorm': self.dmNorm,
            'n_bins_electrons': len(self.energy_bins) - 1,
            'delta_loge_electrons': np.log10(self.energy_bins[1].value) - np.log10(self.energy_bins[0].value),
            'n_bins_nu': len(self.energy_bins_nu) - 1 if self.nuFnu_sim is not None else None,
            'luminosity_GeV_s': self.luminosity.value,
            'luminosity_erg_s': self.luminosity_erg.value,
        }
        pd.Series(metadata).to_csv(output_dir / 'metadata.csv')

    def plot_spectra(self, output_dir: str):
        output_dir = Path(output_dir) / self.cluster_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1, ax1 = plt.subplots(figsize=(9, 8))
        ax1.stairs(self.counts_rate.value, self.energy_bins.value,
                   facecolor="pink", alpha=0.65)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Energy [GeV]", fontsize=20)
        ax1.set_ylabel("Counts rate [1/s]", fontsize=20)
        fig1.savefig(output_dir / 'counts_rate.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(9, 8))
        ax2.stairs(self.dnde_dt.value, self.energy_bins.value,
                   facecolor="gray", fill=True, alpha=0.35)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Energy [GeV]", fontsize=20)
        ax2.set_ylabel(r"$\frac{{\rm d}N}{{\rm d}E {\rm d}t}~[{\rm GeV}^{-1}~{\rm s}^{-1}]$", fontsize=20)
        fig2.savefig(output_dir / 'dNdEdt.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(9, 8))
        ax3.stairs(self.e2_dnde_dt.value, self.energy_bins.value,
                   facecolor="gray", fill=True, alpha=0.35)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Energy [GeV]", fontsize=20)
        ax3.set_ylabel(r"$E^2\frac{{\rm d}N}{{\rm d}E {\rm d}t}~[{\rm GeV}~{\rm s}^{-1}]$", fontsize=20)
        fig3.savefig(output_dir / 'E2dNdEdt.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        if self.nuFnu_sim is not None:
            conv_factor = 1.602176634e-3
            nuFnu_sim_erg = self.nuFnu_sim.value * conv_factor
            nuFnu_prompt_erg = self.nuFnu_prompt.value * conv_factor
            
            interp_total = interp1d(
                self.gamma_engs,
                self.nuFnu_total.value,
                kind='linear',
                bounds_error=False,
                fill_value=0
            )
            nuFnu_total_grid = interp_total(self.bin_centers_nu.value) * conv_factor
            
            fig4, ax4 = plt.subplots(figsize=(13, 10))
            
            ax4.stairs(nuFnu_sim_erg, self.energy_bins_nu.value,
                       facecolor="pink", fill=True, alpha=0.65, label='Secondary')
            
            ax4.plot(self.gamma_engs, nuFnu_prompt_erg,
                     lw=2, color='blue', ls='--', label='Prompt')
            
            ax4.plot(self.bin_centers_nu.value, nuFnu_total_grid,
                     lw=3, color='red', ls='-', label='Total')
            
            ax4.axvspan(xmin=0.02, xmax=300, color='gray', alpha=0.3, label="Fermi-LAT")
            ax4.axvspan(xmin=300, xmax=1e5, color='blue', alpha=0.3, label="LHAASO")
            
            ax4.set_xscale("log")
            ax4.set_yscale("log")
            ax4.set_xlim(1.e-2, 2.e4)
            ax4.set_ylim(1.e-25, 1.e-10)
            ax4.set_xlabel("Energy [GeV]", fontsize=35)
            ax4.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=35)
            ax4.legend(fontsize=26, loc='lower right')
            ax4.tick_params(axis='both', labelsize=24)
            
            fig4.savefig(output_dir / 'nuFnu.png', dpi=150, bbox_inches='tight')
            plt.close(fig4)


def load_clusters_info(clusters_file, name_column='NAME',
                       m200_column='M200',redshift_column='REDSHIFT',
                       distance_column='D_L'):
    
    clusters_file = Path(clusters_file).resolve()
    if not clusters_file.exists():
        raise FileNotFoundError(f"!!! File not found: {clusters_file}")
    
    df = pd.read_csv(clusters_file)
    
    if name_column not in df.columns:
        raise ValueError(f" !!! Column '{name_column}' not found in {clusters_file}")
    
    if m200_column not in df.columns:
        print(f" !!! Column '{m200_column}' not found in {clusters_file}")
        print(f"  Default value:  M200 = 2.40435e14 Msun (for all clusters)")
        df[m200_column] = 2.40435e14
    
    if distance_column not in df.columns and redshift_column in df.columns:
        df[distance_column] = cosmo.luminosity_distance(df[redshift_column].values).value
    
    return df


def parse_arguments():
    parser = argparse.ArgumentParser(description="Luminosity and Spectrum calculation from Photon counts")
    parser.add_argument('--runs-dir', type=str, required=True, help="Simulation runs directory path")
    parser.add_argument('--runs', nargs='+', required=True, help="Names of each runs folders")
    parser.add_argument('--clusters-file', type=str, required=True, help="CSV file with clusters sample info")
    parser.add_argument('--outdir', type=str, default="./dm_analysis_results", help="Output savig directory")
    parser.add_argument('--name-col', type=str, default="NAME", help="Name column label in CSV file")
    parser.add_argument('--m200-col', type=str, default="M200", help="M200 column label in CSV file")
    parser.add_argument('--redshift-col', type=str, default="REDSHIFT", help="Redshift column label in CSV file")
    parser.add_argument('--distance-col', type=str, default="D_L", help="Distance column label in CSV file")
    parser.add_argument('--file-pattern', type=str, default="{cluster}.fits.gz", help="dmspcetra simulation output file pattern for each cluster")
    parser.add_argument('--dm-mass', type=float, default=10000, help="DM mass in GeV")
    parser.add_argument('--process', type=str, default='decay', choices=['decay', 'anna'], help="DM process: decay or anna")
    parser.add_argument('--channel', type=str, default='tau', help="Annihilation/decay channel")
    parser.add_argument('--tau-dm', type=float, default=1e27, help="DM particle lifetime") 
    parser.add_argument('--max-clusters', type=int, default=None, help="Max number of clusters (per run) to analyze")
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--delta-loge', type=float, default=0.2, help="Logarithmic spacing of bins") 
    parser.add_argument('--n-sim', type=int, default=200000, help="Number of simulated particles per run")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    dm_mass = args.dm_mass * u.GeV
    tau_dm = args.tau_dm * u.s
    
    print("=" * 70)
    print("CLUSTER DM ANALYSIS ")
    print("=" * 70)
    
    clusters_df = load_clusters_info(
        args.clusters_file,
        name_column=args.name_col,
        m200_column=args.m200_col,
        redshift_column=args.redshift_col,
        distance_column=args.distance_col
    )
    print(f"  Loaded clusters: {len(clusters_df)}")
    
    if args.max_clusters:
        clusters_df = clusters_df.head(args.max_clusters)
    
    output_dir = Path(args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    luminosity_list = []

    for idx, row in tqdm(clusters_df.iterrows(), total=len(clusters_df), desc="Clusters"):
        cluster_name = row[args.name_col]
        
        try:
            analysis = ClusterAnalysis(
                cluster_name=cluster_name,
                runs_dir=args.runs_dir,
                runs_list=args.runs,
                file_pattern=args.file_pattern,
                dm_mass=dm_mass,
                process=args.process,
                channel=args.channel,
                tau_dm=tau_dm,
                m200=row.get(args.m200_col, None) * u.Msun if args.m200_col in row else None,
                redshift=row.get(args.redshift_col, None),
                distance_luminosity=row.get(args.distance_col, None) * u.Mpc if args.distance_col in row else None,
                n_sim_fixed=args.n_sim
            )
            
            analysis.run_analysis()
            
            analysis.compute_luminosity()
            analysis.save_results(output_dir=output_dir)
            
            if not args.no_plots:
                analysis.plot_spectra(output_dir=output_dir)

            luminosity_list.append({
                'cluster': cluster_name,
                'm200_Msun': analysis.m200.value if analysis.m200 is not None else None,
                'luminosity_GeV_s': analysis.luminosity.value,
                'luminosity_erg_s': analysis.luminosity_erg.value,
            })
            
            results.append({
                'cluster': cluster_name,
                'particles': analysis.total_particles,
                'n_sim_used': args.n_sim if args.n_sim is not None else analysis.total_particles,
                'ndot_exp': analysis.ndot_exp.value if analysis.ndot_exp is not None else None,
                'dmNorm': analysis.dmNorm,
                'luminosity_GeV_s': analysis.luminosity.value,    
                'luminosity_erg_s': analysis.luminosity_erg.value,
            })
            
        except Exception as e:
            print(f"\n  Error processing {cluster_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    luminosity_df = pd.DataFrame(luminosity_list)
    luminosity_df.to_csv(output_dir / 'clusters_luminosities.csv', index=False)
    print(f"\n Luminosities saved in: {output_dir / 'clusters_luminosities.csv'}")
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'analysis_summary.csv', index=False)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED! :)")
    print("=" * 70)
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()