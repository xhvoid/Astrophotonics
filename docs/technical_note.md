# AO-Photonic Spectrograph Performance Simulator

**Independent NIRPS-like, ANDES-inspired conceptual trade study**

> This is an independent conceptual study and is not an official ESO instrument model.

## Abstract

This reproducible Python study connects residual wavefronts, focal-plane complex fields, fibre coupling, detected spectra and radial-velocity (RV) estimation. It asks how atmospheric seeing, residual adaptive-optics (AO) wavefront error, pointing jitter and accepted spatial-mode count influence photon collection and spectral information. A finite-grid scalar model generates phase screens with a von Kármán spatial spectrum, imposes a specified residual optical-path RMS, propagates with a normalised Fourier transform, and projects the complex field onto Gaussian or orthonormal few-mode acceptance fields. A synthetic absorption spectrum then passes through a Gaussian line-spread function and a photon-plus-detector noise model. The design trade is conditional on an explicitly assumed AO attenuation law and ideal modal acceptance. Numerical consistency tests and synthetic injection/recovery establish internal validity; neither instrument-performance validation nor observational RV validation is claimed.

## Motivation

The engineering connection is the propagation of injection losses into an observation's information budget. ESO describes NIRPS as an AO-equipped fibre-fed near-infrared spectrograph. Its historical baseline specifications motivate a 3.6 m aperture, resolving power 100,000 and three detector samples per resolution element in this demonstration. These are selected source-inspired values, not a reconstruction of the installed instrument. [ESO NIRPS](https://www.eso.org/sci/facilities/develop/instruments/NIRPS.html)

ANDES motivates the broader question of AO-fed near-infrared spectroscopy in the ELT era. Its project overview includes diffraction-limited feeding and a near-infrared integral-field capability. Here the phrase *ANDES-inspired* describes that scientific connection; no ELT pupil or ANDES performance budget is implemented. [Marconi et al. 2024](https://arxiv.org/abs/2407.14601)

## Model

### Conventions and provenance

Interfaces state units explicitly: pupil coordinates and optical-path difference (OPD) are in metres, wavelength arguments are in micrometres, residual-WFE arguments are in nanometres, phase is in radians, seeing is in arcseconds at its reference wavelength, jitter is in milliarcseconds, RV is in metres per second, and detector signals are in electrons. WFE is the intensity-weighted spatial RMS of OPD over transmitting pupil samples. Efficiencies and continuum-normalised template fluxes are dimensionless.

| Quantity | Demonstration choice | Status |
|---|---|---|
| Telescope diameter | 3.6 m | Sourced inspiration: ESO NIRPS front-end description |
| Resolving power | 100,000 | Sourced inspiration: ESO NIRPS historical HAM baseline |
| Detector sampling | 3 pixels per LSF FWHM | Sourced inspiration: same ESO baseline table |
| Spectral interval | 1540–1570 nm | Assumed narrow near-infrared interval |
| Seeing reference | 0.8 arcsec at 500 nm | Assumed AO-reference condition |
| Pupil obscuration and spiders | Configurable geometry | Assumed; not a measured telescope mask |
| WFE, jitter and outer scale | 150 nm, 10 mas per axis, 25 m | Assumed residual-error scenario |
| Gaussian field radius | 57 mas | Assumed injection matching, approximately $0.64\lambda_\mathrm{ref}/D$ at 1550 nm |
| Modal capacities | 1, 3, 6, 10 and 15 spatial modes | Illustrative design alternatives |
| Lantern transmission | 0.9 | Assumed, independent of mode and wavelength |
| Other throughput, exposure, source flux and detector noise | Configuration-driven | Assumed, not measured instrument properties |
| Stellar lines and injected RV | Synthetic demonstration | Illustrative; no fitted astrophysical parameters |

The executable baseline configuration is the authority for a particular run. No model parameter has been fitted to NIRPS, ANDES or laboratory data.

### Pupil, turbulence and residual AO error

A binary circular aperture can include a concentric obscuration and simplified straight spiders. For the conventional Kolmogorov seeing parameter,

$$
r_0(\lambda_0)=\frac{0.98\lambda_0}{\epsilon_0},\qquad
r_0(\lambda)=r_0(\lambda_0)(\lambda/\lambda_0)^{6/5},
$$

where seeing $\epsilon_0$ is converted to radians. This convention labels turbulence strength; with finite outer scale, the actual long-exposure image FWHM need not equal that seeing value. [Fried 1966](https://doi.org/10.1364/JOSA.56.001372), [Tokovinin 2002, Eq. 5](https://www.ctio.noirlab.edu/~atokovin/papers/pasp2002.pdf)

With radial spatial frequency $f$ in cycles m$^{-1}$, the screen-shape spectrum is

$$
W_\phi(f)=0.023r_0^{-5/3}(f^2+L_0^{-2})^{-11/6}.
$$

Fourier-domain Gaussian random coefficients generate reproducible real-valued screens. The DC component is removed. A finite sampled screen omits turbulence scales beyond its extent and below its pixels; it is not an exact atmospheric covariance model. [Fétick et al. 2018, Eq. 17](https://doi.org/10.1093/mnras/sty2595)

The AO stage is a reduced residual-error prescription. A weighted least-squares fit removes piston and both tilt terms; the remaining OPD receives one scalar rescaling to the requested pupil RMS. All remaining spatial orders receive the same attenuation: no AO spatial transfer function is applied. The configured reference residual $\sigma_{W,\mathrm{ref}}$ is defined at seeing $\epsilon_\mathrm{ref}$; for a seeing sweep at fixed assumed AO attenuation,

$$
\sigma_W(\epsilon)=\sigma_{W,\mathrm{ref}}
(\epsilon/\epsilon_\mathrm{ref})^{5/6}.
$$

This scaling follows the square root of the $r_0^{-5/3}$ variance dependence. It is an imposed sensitivity law, not a prediction of a wavefront sensor, deformable mirror or controller. Enforcing exact RMS on each realisation intentionally removes the natural distribution of screen RMS. Consequently, when actual residual RMS is held fixed, this model provides no additional independent seeing penalty.

Pointing offset is added through the pupil phase ramp $2\pi(x\alpha_x+y\alpha_y)/\lambda$. Jitter denotes the Gaussian RMS **per Cartesian axis**; radial RMS is $\sqrt{2}$ times that value. Independent realisations approximate an ensemble, not a temporal AO simulation.

### Fraunhofer field and image metrics

For transmission $P$ and residual OPD $W$,

$$
E_\lambda(\boldsymbol\theta)\propto
\mathcal{F}\{P(\mathbf{x})\exp[2\pi iW(\mathbf{x})/\lambda]\}
(\boldsymbol\theta/\lambda),\qquad I_\lambda=|E_\lambda|^2.
$$

The orthonormal FFT preserves the discrete squared field norm. The angular spacing is $\Delta\theta=\lambda/(N_\mathrm{FFT}\Delta x)$; zero-padding improves focal-plane sampling without adding pupil information. Normalised PSFs sum to one, while the collecting-area and photon budget are carried separately. This prevents a change in FFT size from changing the detected flux. [Goodman 2017](https://www.macmillanlearning.com/ed/uk/product/Introduction-to-Fourier-Optics-4th-edition/p/1319119166)

Strehl is the ratio of the sampled maximum intensities of independently flux-normalised aberrated and ideal images made from the **same pupil and sampling**. The baseline reports the peak of the ensemble-mean intensity, so its jitter blur remains present. For a small phase error without that blur, the extended Maréchal approximation is

$$
S_\mathrm{M}=\exp[-(2\pi\sigma_W/\lambda)^2].
$$

It is a cross-check in the high-Strehl regime, not a replacement for propagation at large WFE. A shifted image can retain a high peak while coupling poorly to a stationary fibre; peak Strehl and pointing loss must be interpreted separately. [Mahajan 1983](https://doi.org/10.1364/JOSA.73.000860)

Limited polychromatic images combine **intensities** on a common angular grid using flux-conserving pixel-overlap resampling. The default grid encloses the component images; a cropped grid retains its missing flux rather than renormalising it away. Fields at different optical wavelengths are not coherently summed. Coupling is evaluated from each wavelength's complex field before averaging; the square root of a long-exposure PSF is not a valid substitute for that field. The spectral pipeline samples mean coupling at three wavelength nodes and interpolates it across the narrow demonstration band.

### Fibre and ideal lantern acceptance

The Gaussian amplitude is $u_{00}\propto\exp[-\theta^2/w^2]$, so $w$ is its $1/e$ amplitude radius and $1/e^2$ intensity radius. The corresponding angular mode-field diameter is $2w$. The configured angular scale represents injection optics as well as the fibre; it is not a physical core diameter without a specified focal length.

SMF coupling uses the normalised complex overlap,

$$
\eta_1=\frac{|\langle E,u_{00}\rangle|^2}
{\langle E,E\rangle\langle u_{00},u_{00}\rangle}.
$$

The field phase and uniform-grid area factors are retained consistently. Cauchy–Schwarz bounds efficiency between zero and one. An ideal Airy field need not couple with unit efficiency because its amplitude differs from a Gaussian. [Shaklan and Roddier 1988](https://doi.org/10.1364/AO.27.002334)

For the reduced few-mode model, Hermite–Gaussian fields ordered by total transverse degree provide a nested, numerically orthonormal basis. It represents idealised scalar parabolic-index acceptance, not a measured step-index fibre. Mode groups through degrees 0, 1, 2, 3 and 4 contain 1, 3, 6, 10 and 15 spatial modes. With $\langle u_j,u_k\rangle=\delta_{jk}$,

$$
\eta_N=\frac{\sum_{j=1}^{N}|\langle E,u_j\rangle|^2}{\langle E,E\rangle},
\qquad \eta_{\mathrm{PL},N}=T_\mathrm{PL}\eta_N.
$$

Bessel's inequality guarantees $\eta_N\leq1$; nested subspaces guarantee that accepted power cannot decrease when modes are added. A lantern then maps this accepted subspace into enough single-mode outputs, with assumed scalar transmission. The relevant lantern principle is conservation of modal capacity in an adiabatic transition. Summed power needs no arbitrary random transfer matrix. Polarisation, individual-port fluxes, taper design and measured mode-dependent losses are not predicted. [Birks et al. 2015](https://doi.org/10.1364/AOP.7.000107)

The few-mode comparison treats its accepted light as one extracted spectrum. A lantern has one independently read spectral trace per port. This is an explicit detector-accounting assumption: at equal modal acceptance, a lantern cannot create photons, and extra traces introduce extra read/background variance. Real formats require a separate detector-layout study.

### Spectrum, throughput and detector

A continuum-normalised synthetic absorption template is Doppler shifted and convolved with a Gaussian LSF. Constant resolving power is represented on a log-wavelength grid; $\mathrm{FWHM}_v=c/R$, followed by detector sampling. The narrow default interval keeps the calculation tractable and does not reproduce an entire echelle order or stellar atmosphere. Imported one-dimensional templates are an extension of the same interface.

Writing $f_i$ for the normalised sampled spectrum, the expected source signal has the form

$$
\mu_i=F_{\lambda,\mathrm{cont}}\,A_\mathrm{clear}\,t\,
\Delta\lambda_i\,T_\mathrm{other}\,\eta(\lambda_i)\,f_i.
$$

Source continuum photon density is specified in photons s$^{-1}$ m$^{-2}$ µm$^{-1}$ before telescope collection. Throughput includes optical transmission and detector quantum efficiency. Coupling appears once. Collecting area is computed from $\sum P^2\Delta x^2$, so sampled obscuration and spiders remove photons consistently with the pupil mask.

For $n_\mathrm{ch}$ extracted channels, $n_\mathrm{ext}$ detector pixels per channel and spectral bin, per-pixel background/dark rate $b+d$, and per-pixel read noise $r$, the variance model is

$$
V_i=\mu_i+n_\mathrm{ch}n_\mathrm{ext}(b+d)t+n_\mathrm{ch}n_\mathrm{ext}r^2,
\qquad \mathrm{SNR}_i=\mu_i/\sqrt{V_i}.
$$

Poisson fluctuations are drawn for source and background/dark electrons; Gaussian read noise is added. Subtracting a known mean background removes its offset but not its shot noise. This assumes independent pixels and known background, not an additional noisy sky exposure.

### Radial-velocity recovery

A known RV is injected into the template with wavelength factor $\sqrt{(1+v/c)/(1-v/c)}$; positive velocity denotes redshift. Recovery minimises a variance-weighted $\chi^2$ using the same forward model, a known continuum and known expected variance. This isolates numerical recovery and information loss; matching injection and retrieval is deliberately an internally consistent synthetic test. It cannot assess template mismatch or calibration drift. Very low photon counts would require a Poisson likelihood instead of this Gaussian approximation.

The local spectral Fisher estimate is

$$
\sigma_v\simeq\left[\sum_i V_i^{-1}
\left(\frac{\partial\mu_i}{\partial v}\right)^2\right]^{-1/2}.
$$

This is the spectral-gradient RV information argument with the implemented detector variance. At fixed spectrum and photon-limited throughput it predicts $\sigma_v\propto\eta^{-1/2}$. It is a conditional precision estimate, not an instrumental accuracy floor. A single noisy recovery bias is a random draw; an ensemble is required to establish estimator bias. [Bouchy, Pepe and Queloz 2001](https://doi.org/10.1051/0004-6361:20010730)

## Validation

Validation separates physical invariants, numerical checks and statistical behaviour. The automated suite covers the unobscured Airy reference, FFT energy conservation, unit diffraction-limited Strehl, high-Strehl agreement with the extended Maréchal approximation, bounded modal coupling, seed reproducibility, severe-aberration ensemble behaviour, noiseless RV injection/recovery and the full baseline pipeline. Tests state the sampling or statistical reason for important tolerances; machine precision is appropriate for Parseval's identity but not for pixelised Airy-profile agreement or finite ensembles.

The executed showcase notebook provides visual checks of OPD, diffraction structure, coupling trends, spectral sampling and the RV likelihood. Exact test counts, runtime and reproduced numerical results are recorded with the completed execution. These checks validate the implemented approximation and its numerical chain. They do not establish correct atmospheric temporal covariance, a real lantern's mode mapping or an instrument's achieved RV stability.

## Results

<!-- BEGIN GENERATED RESULTS -->
**Validation: 78 tests passed; all 12 notebook code cells executed without errors; 9 PNG/PDF figure pairs exported.** The last complete notebook execution took 19.9 seconds on the validation machine (Python 3.14.3; CPU only). Runtime varies by machine.

| Baseline quantity | Synthetic result |
|---|---:|
| Residual WFE | 150 nm |
| Measured ensemble Strehl (with jitter) | 0.6598 |
| Maréchal cross-check (without jitter) | 0.6926 |
| Mean SMF coupling across band | 0.4711 |
| Mean ideal six-mode acceptance | 0.5303 |
| Mean six-port lantern coupling, after 90% transmission | 0.4772 |
| Median SNR per detector pixel | 144.06 |
| Injected RV | 2300.00 m/s |
| Recovered RV | 2296.37 m/s |
| Recovery error | -3.63 m/s |
| Conditional RV uncertainty (1σ) | 4.12 m/s |
| RV fit reduced χ² (≈1 when model and variance describe the data) | 0.962 |

1. At zero jitter, increasing residual WFE from 0 to 500 nm changes mean SMF coupling from 0.699 to 0.020; wavefront quality sets the photon budget.

2. At 150 nm WFE, increasing per-axis jitter from 0 to 40 mas changes SMF coupling from 0.489 to 0.249; injection stability remains important even at fixed AO residual.

3. At 400 nm WFE / 10 mas jitter, replacing the SMF with a 15-port lantern changes forecast σRV from 12.98 to 8.75 m/s at baseline brightness, but from 403.84 to 502.86 m/s at 1/100 brightness. Under the assumed insertion loss and extraction noise, detector costs can reverse the throughput advantage.

The design map uses 12 independent optical realizations per cell and common random inputs between alternatives. Its largest SMF standard error of the mean is 0.064; uncertainty bars are shown in the pointing plot, and the map saves each standard error. Port-count rankings close to each other should not be interpreted as established optima. In particular, the 150 nm bright case is a nominal near tie between the SMF and 15-port model.

The no-jitter 150 nm illustrative PSF has measured Strehl 0.6947, compared with Maréchal 0.6926. At 400 nm the same wavefront shape gives 0.2135 versus 0.0734, demonstrating why the approximation is not imposed at high aberration. The baseline instead averages independent PSFs and includes jitter, explaining its different measured Strehl.

Numbers come from [showcase_metrics.json](../results/showcase_metrics.json), [design_map.npz](../results/design_map.npz), [port_trade.json](../results/port_trade.json) and the [executed notebook](../notebooks/eso_astrophotonics_showcase.ipynb). Regenerate these artifacts before running `scripts/update_result_sections.py`.
<!-- END GENERATED RESULTS -->

## Limitations

- The circular pupil, obscuration and straight spiders are conceptual geometry. No segmented ELT pupil, coronagraph or full optical prescription is included.
- Screens have finite spatial extent, and the residual AO prescription fixes their RMS. There is no atmospheric layer/wind model, scintillation, anisoplanatism, wavefront-sensor noise, deformable-mirror dynamics or closed-loop controller. Jitter realisations have no temporal power spectrum.
- Gaussian/Hermite–Gaussian acceptance omits real fibre index profiles, chromatic cutoffs, polarisation, bends, focal-ratio degradation, mode-dependent loss, manufacturing errors and modal-noise-induced LSF changes. Port count is a chosen spatial capacity, not inferred from a core radius and numerical aperture.
- The detector is a one-dimensional extraction model with fixed read noise and independent pixels. Persistence, nonlinearity, correlated noise, defective pixels, saturation and extraction cross-talk are absent.
- The narrow synthetic spectrum excludes tellurics, stellar activity, line-list uncertainty, rotation-dependent spectral diversity, barycentric corrections, wavelength-solution errors and instrument drift. It cannot support claims of metre-per-second performance for NIRPS or ANDES.
- OOPAO is an adapter/extension target only. No OOPAO integration or validation has been completed. Public observational validation remains future work.

## Future experimental validation

A realistic first experiment would illuminate a scaled pupil with a narrowband fibre-fed source, impose known phase patterns with a deformable mirror or spatial light modulator, and measure total incident and transmitted power while scanning injection angle. Camera images and calibrated wavefront measurements would independently constrain PSF sampling, OPD RMS and Gaussian mode radius. Repeating the measurement with an SMF, a few-mode fibre and a characterised lantern would reveal where ideal modal projection fails.

Next, disperse each output with a stable line source. Measure channel-dependent throughput, read/background variance and the LSF before testing RV-like shifts against a common wavelength reference. Record uncertainties, the exact fibre/lantern serial identifiers and the bench's plate scale. A held-out set of phase patterns and wavelengths should test the calibrated model. This proposed workflow is aligned with the optomechanical and analysis tasks described in ESO's photonic-chip testbench topic; it has not been performed here. [ESO testbench topic](https://www.eso.org/sci/activities/fellowships-and-studentships/engstudentfellow/photonic_chips.html)

Public NIRPS spectra could subsequently test wavelength sampling, continuum SNR and template-based RV extraction after file provenance, reduction version and quality flags are recorded. They would validate the spectroscopic portion only; without simultaneous wavefront and injection telemetry they cannot directly validate this model's AO-to-coupling map.

## References

Full bibliographic details, equation provenance, instrument-source caveats and software acknowledgements are maintained in [references.md](references.md). Core numerical work uses NumPy and SciPy, visualisation uses Matplotlib, and astronomical file support uses Astropy. These dependencies and OOPAO's separately scoped extension role are cited there.
