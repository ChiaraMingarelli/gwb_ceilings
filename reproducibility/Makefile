PYTHON ?= python
SOURCE_DATE_EPOCH ?= 946684800
MPLCONFIGDIR ?= $(CURDIR)/.mplconfig
export SOURCE_DATE_EPOCH
export MPLCONFIGDIR

.PHONY: all ledger fmerge test figures check-generated verify \
	check-paper-assets sync-paper-assets manifest check-manifest \
	release-check public-release-check

all: verify figures

ledger:
	$(PYTHON) verify_ceilings.py \
		--json results/step3-numbers.json \
		--npz results/step3-spectra.npz

fmerge:
	$(PYTHON) fmerge/lm24_cnm_posterior.py \
		--samples 262144 \
		--seed 20240724 \
		--output results/fmerge/summary.json \
		--sample-output results/fmerge/posterior_draws.npz

test:
	$(PYTHON) -m unittest -v \
		tests/test_verify_ceilings.py \
		tests/test_figure_context.py
	$(PYTHON) -m unittest -v fmerge/test_lm24_cnm_posterior.py

figures:
	$(PYTHON) make_figure1_plane_forest.py
	$(PYTHON) fmerge/plot_fmerge_posterior.py
	$(PYTHON) make_figure3_multiband.py

check-generated:
	$(PYTHON) tools/check_generated.py

verify: test check-generated release-check

check-paper-assets:
	$(PYTHON) tools/sync_manuscript_assets.py

sync-paper-assets:
	$(PYTHON) tools/sync_manuscript_assets.py --write

manifest:
	$(PYTHON) tools/write_manifest.py --write

check-manifest:
	$(PYTHON) tools/write_manifest.py --check

release-check:
	$(PYTHON) tools/release_check.py
	$(PYTHON) tools/write_manifest.py --check

public-release-check:
	$(PYTHON) tools/release_check.py --require-license
	$(PYTHON) tools/write_manifest.py --check
