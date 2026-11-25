# Figures for the augmented iterative method paper.
#
#   make figures          regenerate every figure from the archived results
#   make fetch            download that archive (set URL=... or PRESTRESS_DATA_URL)
#   make archive          rebuild the archive from a local output/ tree
#   make test             the checks that do not need FEBio
#
# "make figures" needs no FEBio, no cluster and none of the source geometry:
# each .vtu carries its own mesh, displacement, principal stretches and
# directions.  See docs/data_release.md.

PYTHON ?= python3
DATA   ?= data
FIGDIR ?= figures
URL    ?=

CASES := cylinder aorta heart

.PHONY: figures fetch archive test clean help

help:
	@sed -n '3,6p' Makefile

fetch:
	$(PYTHON) scripts/fetch_results.py $(if $(URL),--url $(URL),) --dest $(DATA)

figures:
	@mkdir -p $(FIGDIR)
	@for c in $(CASES); do \
		echo "--- $$c ---"; \
		$(PYTHON) scripts/visualize_prestress.py \
			--from-vtu $(DATA)/vtu/$$c.vtu \
			--out $(FIGDIR)/prestress_$$c || exit 1; \
	done
	$(PYTHON) scripts/plot_convergence.py --root $(DATA) \
		--out $(FIGDIR)/convergence.png
	@echo "\nfigures in $(FIGDIR)/"

archive:
	$(PYTHON) scripts/package_results.py

test:
	$(PYTHON) tests/test_basics.py

clean:
	rm -rf $(FIGDIR) dist
