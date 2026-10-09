# Public spectra: import boundary and future validation

**Status: synthetic validation only.** No public NIRPS spectrum has been downloaded or analysed in this project. The baseline and showcase use generated absorption lines. Import support does not establish observational validation.

## Finding one small public product

ESO publishes a NIRPS reduced-spectra collection under [DOI 10.18727/archive/92](https://doi.eso.org/10.18727/archive/92). Its [release announcement](https://archive.eso.org/cms/eso-archive-news/release-of-nirps-pipeline-reduced-spectra.html) links to the interactive Science Portal and the release documentation. The collection is dynamic, so its DOI alone does not identify an individual reproducible observation.

For a future validation experiment:

1. Open the [ESO Archive Science Portal](https://archive.eso.org/scienceportal/home), select NIRPS reduced spectra, and restrict the query to one public target/exposure with suitable near-infrared coverage. Inspect the listed product size before downloading. Avoid an unrestricted collection download.
2. Read the product preview, quality information and [release description](https://www.eso.org/rm/api/v1/public/releaseDescriptions/233). Prefer the primary target-fibre one-dimensional product for a first sampling check. The release identifies `S1D_FINAL_A` as a primary product, and provides associated CCF and optional ancillary products. Order-by-order `S2D` arrays require a separate extraction or order-selection step.
3. Save the selected product unchanged in the ignored local `data/` directory. Record the archive's exact file/product identifier and download URL, original filename, target, observing date, exposure, observing mode, programme identifier, reduction version, access date and a SHA-256 checksum. Do not substitute an invented example identifier.
4. Inspect the actual FITS headers and table schema. Record the wavelength unit, air/vacuum convention, observer/barycentric/rest frame, applied corrections, flux/error units and quality-flag definitions. Choose the wavelength array deliberately; no convention is inferred from the numerical scale alone.
5. Create a separate small analysis using the product's supplied error and quality arrays. Start with wavelength spacing and continuum-region SNR. Preserve the original data and document every selection, continuum fit and resampling operation in a transformation record.
6. Report this observational result separately from the synthetic showcase, including the exact file identifier and limitations. Follow the collection's acknowledgement and reference instructions. Only claim a successful data comparison after its notebook or script has run reproducibly.

Keep a provenance record such as the following **unfilled schema** next to any future analysis. It is not a record of downloaded data:

```yaml
collection_doi: "10.18727/archive/92"
product_identifier: null
source_url: null
original_filename: null
sha256: null
accessed_utc: null
target: null
observation_date_utc: null
programme_id: null
instrument_mode: null
exposure_s: null
pipeline_name_version: null
wavelength_unit: null
wavelength_medium: null
velocity_reference_frame: null
applied_corrections: []
flux_unit: null
error_column: null
quality_flag_definition: null
transformations: []
```

## Exact template formats accepted by this package

`aophoton.spectrum.load_spectrum(path)` reads a **dimensionless relative-flux template**, not a general observational data product. It returns a `Spectrum` containing `wavelength_um` and `relative_flux`. Each array must be one-dimensional with matching length of at least three. Wavelengths must be positive and strictly increasing; all entries must be finite; template flux must be nonnegative. Negative noisy samples must not be silently clipped merely to make an observed spectrum into a template.

| Format | Accepted schema |
|---|---|
| ASCII | Exactly two numeric columns: wavelength in µm, then dimensionless relative flux. `.csv` files use commas; other non-FITS files use whitespace. Use `#` for optional comment/header lines. |
| FITS binary table | Scalar-sample columns named exactly `WAVELENGTH` and `FLUX`. The wavelength column must have an explicit `TUNIT` convertible to length in µm. Flux units must be dimensionless or absent. |
| FITS primary image | A one-dimensional flux array and a linear wavelength axis with `CTYPE1='WAVE'`, `CRVAL1`, `CDELT1`, `CRPIX1`, and length-valued `CUNIT1`. `BUNIT` must be dimensionless or absent. |

FITS import requires the optional dependency:

```bash
python -m pip install -e '.[fits]'
```

No conversion is made from flux-calibrated physical units to a normalised template. No quality mask, error array, barycentric correction or rest-frame conversion is applied. No echelle order merging is performed. In particular, a native ESO Phase 3 table may store an entire array in one row; such a vector-valued column is **not** the scalar-row schema accepted here. It needs an explicit, product-aware conversion. The [ESO reduced-product description](https://archive.eso.org/cms/eso-data/phase3/ESO_reduced_data_products_description.pdf) explains the vector-column format and wavelength-convention metadata.

After an external conversion has produced a suitable template, import it directly:

```python
from aophoton.spectrum import load_spectrum

template = load_spectrum("data/normalized_template.csv")
```

To use that file in the forward simulation, copy `configs/baseline.yaml` to another file in `configs/`, then set:

```yaml
observation:
  template_path: ../data/normalized_template.csv
```

Keep the other desired observation settings in the same mapping. Paths are resolved relative to the YAML file. The spectral model verifies coverage for both RV extremes and LSF padding; the template must extend beyond the plotted interval. The default synthetic support, 1.535–1.575 µm, is comfortably wider than the default 1.540–1.570 µm observation, but imported templates should satisfy the actual bounds reported by `SpectrographModel`. Dense template sampling must also resolve the line profiles; passing the format check alone is not sufficient.

## What can be validated

Wavelength sampling and product SNR are useful first comparisons because they can be measured without adopting this simulator's AO model. Use the documented observation mode and actual product LSF. The [operational NIRPS collection description](https://doi.eso.org/10.18727/archive/92) lists median resolving powers of about 82,000 in HA and 75,000 in HE; this repository's 100,000 baseline is historical design inspiration.

An observed template already contains its instrument's broadening and noise. Passing it through `SpectrographModel` adds another Gaussian LSF, so that operation is a new synthetic degradation experiment, not a direct reproduction of the original exposure. Avoid claiming an absolute stellar RV from a template whose rest-frame zero point is unknown.

The low-level `recover_rv` interface accepts observed flux and variance plus a caller-supplied spectral model, but does not implement continuum fitting, velocity-frame conversion or template uncertainties. An observational RV study therefore needs a separate likelihood model and provenance-aware preparation. Comparison with a supplied pipeline CCF must use the same frame and sign convention and account for different masks and telluric handling.

A public spectrum without simultaneous wavefront, jitter and injection telemetry cannot validate the atmospheric/AO-to-coupling map. That part requires an optical testbench or appropriately documented telemetry. No observational file identifiers, measured SNR values or recovered real-star RVs are supplied here because that work remains future validation.
