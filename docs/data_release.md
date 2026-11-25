# Releasing the results so a reader can regenerate the figures

The problem is not size, it is dependencies. Every figure in the paper is
three steps downstream of something a reader cannot easily get: a 337 MB VMR
model and a 35 MB heart mesh that belong to other people, a patched FEBio
build, and several hours on a cluster. A reader who wants to check that
`lambda_1` really is circumferential should not have to reproduce any of that.

So the release is layered, and the layer that makes the figures carries no
dependencies at all.

## Three layers

| layer | what | size | what it needs |
|---|---|---|---|
| **figures** | one `.vtu` per geometry + the ladders as JSON | 50 MB packed | python, numpy, pyvista, matplotlib |
| **simulations** | the AIM results as `.npz` (reference configuration, residual history) | 8.5 MB | the above, plus the source geometry |
| **inputs** | the VMR model, the Pfaller19 heart mesh | 372 MB | *not redistributed* -- fetched from their own records |

Each `.vtu` is self-describing: mesh, connectivity, nodal displacement, the
three principal stretches, their directions, the exclusion mask, and the case
name and title in field data. `visualize_prestress.py --from-vtu` reads one
and draws every panel without importing a solver, rebuilding a wall or
touching the source geometry. That is what makes the regeneration one step
instead of a project.

```bash
make fetch URL=https://zenodo.org/records/<id>/files/prestress-figures.tar.gz
make figures
```

`fetch_results.py` verifies every file against the SHA-256 manifest packed
with the archive, and extracts with `filter="data"` because the tarball is
remote and therefore untrusted.

## Where to put it

**Zenodo**, as a record tied to the paper. The reasons, in order:

* It mints a **DOI**, so the archive is citable in the data-availability
  statement rather than being a URL that rots.
* It is **permanent and versioned** -- a v2 gets its own DOI and the concept
  DOI always resolves to the latest.
* **50 GB per record**, against our 50 MB, so there is no reason to prune.
* It **hooks onto GitHub releases**: tag the repository, and Zenodo archives
  the code at that tag beside the data, which pins the exact script that drew
  each figure.

Alternatives considered and rejected: a GitHub release alone (no DOI, no
guarantee past the repository's life), institutional storage (no DOI, access
usually tied to the group), and committing the data (50 MB of binaries in git
history, and it still would not survive a repository move).

## One click, for real

A reader clicking a link in the PDF should see the figures, not a shell.
`notebooks/regenerate_figures.ipynb` is a Colab notebook that installs the
dependencies, fetches the archive, regenerates every panel and displays them
inline. Put the badge in the README and in the data-availability statement:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/Prestress/blob/main/notebooks/regenerate_figures.ipynb)
```

Colab gives a free GPU-less VM with enough memory for the 167k-element heart,
and pyvista renders off-screen there, so nothing has to be installed locally.

## Licensing

The `.vtu` and `.npz` files are **derived** data -- our solution fields on our
own wall mesh -- and go out under CC BY 4.0 with the code under its existing
licence. The inputs do not travel with them:

* the aorta is **VMR model `0012_H_AO_H`**, from the Vascular Model Repository
  (vascularmodel.com); cite the repository and SimVascular as they ask.
* the heart is `eva.vtu` from **Pfaller et al., Biomech Model Mechanobiol 18
  (2019) 503-529**, *The importance of the pericardium for cardiac
  biomechanics*.

`scripts/fetch_results.py` takes a URL rather than hard-coding one, so a
mirror or an institutional copy works without editing the code.

## Steps to publish

1. `make archive` -- writes `dist/prestress-figures.tar.gz` and
   `dist/MANIFEST.json`.
2. Create the Zenodo record, upload both, set the licence and the authors,
   and reserve the DOI.
3. Put the DOI in the README, in `docs/data_release.md`, and in the paper's
   data-availability statement.
4. Enable the Zenodo-GitHub hook and tag a release, so the code is archived at
   the same moment.
5. Check the Colab badge end to end from a logged-out browser. A link that
   only works for the author is the usual way this fails.
