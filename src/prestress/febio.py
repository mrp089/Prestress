"""Write, run and read FEBio 4 models of the prestressed cylinder.

Two material models are used:

``Mooney-Rivlin``
    FEBio's built-in uncoupled solid with ``c2 = 0``, i.e. exactly Eq. 11 of the
    manuscript.  Used for the black-box AIM solves.

``prestress_cmm``
    The plugin in ``febio_plugin/``: the same elastin model plus one family of
    embedded fibres that can turn over.  Used for the growth study, and, with
    ``phi_c = 0``, as a drop-in replacement for the elastin-only model so that
    both stages of the study use an identical discretisation.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .mesh import CylinderMesh
from .params import MaterialParams

__all__ = [
    "febio_executable",
    "plugin_path",
    "write_elastin_feb",
    "write_cmm_feb",
    "run_febio",
    "read_node_log",
    "read_node_log_all",
]

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FEBIO = _ROOT / "external/FEBio/cbuild/bin/febio4"
_PLUGIN = _ROOT / "febio_plugin/prestress_cmm.dylib"
_PLUGIN_SO = _ROOT / "febio_plugin/prestress_cmm.so"


def febio_executable() -> str:
    """Locate the FEBio executable (``FEBIO_EXE`` overrides)."""
    exe = os.environ.get("FEBIO_EXE")
    if exe:
        return exe
    if _DEFAULT_FEBIO.is_file():
        return str(_DEFAULT_FEBIO)
    found = shutil.which("febio4") or shutil.which("febio3")
    if found:
        return found
    raise FileNotFoundError(
        "FEBio not found; build it under external/FEBio or set FEBIO_EXE"
    )


def plugin_path() -> str:
    """Locate the compiled ``prestress_cmm`` plugin (``FEBIO_PLUGIN`` overrides)."""
    p = os.environ.get("FEBIO_PLUGIN")
    if p:
        return p
    for cand in (_PLUGIN, _PLUGIN_SO):
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "prestress_cmm plugin not found; run febio_plugin/build.sh"
    )


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
_SOLVER = """\
    <solver>
      <symmetric_stiffness>0</symmetric_stiffness>
      <dtol>0</dtol>
      <etol>{etol}</etol>
      <rtol>{rtol}</rtol>
      <lstol>0.9</lstol>
      <min_residual>{min_residual}</min_residual>
      <max_refs>{max_refs}</max_refs>
      <qn_method type="BFGS">
        <max_ups>0</max_ups>
      </qn_method>
      <reform_each_time_step>1</reform_each_time_step>
      <diverge_reform>1</diverge_reform>
    </solver>
"""


def _fmt_nodes(nodes):
    return "\n".join(
        f'      <node id="{i + 1}">{x:.16e},{y:.16e},{z:.16e}</node>'
        for i, (x, y, z) in enumerate(nodes)
    )


def _fmt_elems(hexes):
    return "\n".join(
        f'      <elem id="{i + 1}">' + ",".join(str(n + 1) for n in e) + "</elem>"
        for i, e in enumerate(hexes)
    )


def _fmt_facets(facets):
    return "\n".join(
        f'      <quad4 id="{i + 1}">' + ",".join(str(n + 1) for n in f) + "</quad4>"
        for i, f in enumerate(facets)
    )


def _fmt_nodeset(ids):
    return "      " + ",".join(str(i) for i in np.unique(ids) + 1)


def _fmt_domain_opts(laugon: bool, augtol: float, maxaug: int) -> str:
    """Augmented-Lagrangian settings for the three-field solid domain.

    With ``laugon`` the incompressibility constraint is enforced through a
    Lagrange multiplier iterated to ``augtol``, instead of only through the
    bulk penalty, so the solution no longer carries an O(mu/K) error.
    """
    if not laugon:
        return ""
    return (
        "      <laugon>1</laugon>\n"
        f"      <atol>{augtol:.12g}</atol>\n"
        "      <minaug>1</minaug>\n"
        f"      <maxaug>{maxaug}</maxaug>\n"
    )


def _fmt_node_map(name: str, values: np.ndarray) -> str:
    """Per-node vec3d map, ``values`` of shape (n_node, 3)."""
    rows = "\n".join(
        f'      <node lid="{i + 1}">' + ",".join(f"{v:.17g}" for v in row) + "</node>"
        for i, row in enumerate(values)
    )
    return (
        f'    <NodeData name="{name}" data_type="vec3" node_set="all">\n'
        f"{rows}\n    </NodeData>"
    )


def _fmt_mp_map(name: str, values: np.ndarray) -> str:
    """Per-integration-point scalar map, ``values`` of shape (n_elem, n_gp)."""
    rows = "\n".join(
        f'      <e lid="{i + 1}">' + ",".join(f"{v:.17g}" for v in row) + "</e>"
        for i, row in enumerate(values)
    )
    return (
        f'    <ElementData name="{name}" data_type="scalar" '
        f'format="MAT_POINTS" elem_set="wall">\n{rows}\n    </ElementData>'
    )


# ---------------------------------------------------------------------------
# model writers
# ---------------------------------------------------------------------------
def _assemble(
    path: Path,
    mesh: CylinderMesh,
    coords: np.ndarray,
    material: str,
    pressure: float,
    control: str,
    load_points: str,
    meshdata: str = "",
    domain_opts: str = "",
    extra_nodesets: str = "",
    extra_loads: str = "",
    plot: bool = False,
    plot_vars: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotfile = (
        f'    <plotfile type="vtk">\n{plot_vars}    </plotfile>\n' if plot else ""
    )
    meshdata_block = f"  <MeshData>\n{meshdata}\n  </MeshData>\n" if meshdata else ""

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
{control}\
  </Control>
  <Material>
{material}\
  </Material>
  <Mesh>
    <Nodes name="vessel">
{_fmt_nodes(coords)}
    </Nodes>
    <Elements type="hex8" name="wall">
{_fmt_elems(mesh.hexes)}
    </Elements>
    <Surface name="lumen">
{_fmt_facets(mesh.inner_facets)}
    </Surface>
    <NodeSet name="fix_x">
{_fmt_nodeset(mesh.fix_x)}
    </NodeSet>
    <NodeSet name="fix_y">
{_fmt_nodeset(mesh.fix_y)}
    </NodeSet>
    <NodeSet name="fix_z">
{_fmt_nodeset(mesh.fix_z)}
    </NodeSet>
{extra_nodesets}\
  </Mesh>
  <MeshDomains>
    <SolidDomain name="wall" mat="wall_material">
{domain_opts}\
    </SolidDomain>
  </MeshDomains>
{meshdata_block}\
  <Boundary>
    <bc name="bcx" type="zero displacement" node_set="fix_x">
      <x_dof>1</x_dof><y_dof>0</y_dof><z_dof>0</z_dof>
    </bc>
    <bc name="bcy" type="zero displacement" node_set="fix_y">
      <x_dof>0</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>
    </bc>
    <bc name="bcz" type="zero displacement" node_set="fix_z">
      <x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <Loads>
    <surface_load name="luminal_pressure" type="pressure" surface="lumen">
      <pressure lc="1">{pressure:.16g}</pressure>
      <linear>0</linear>
      <symmetric_stiffness>0</symmetric_stiffness>
    </surface_load>
{extra_loads}\
  </Loads>
  <LoadData>
    <load_controller id="1" name="ramp" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <points>
{load_points}
      </points>
    </load_controller>
  </LoadData>
  <Output>
{plotfile}\
    <logfile>
      <node_data data="x;y;z" file="{path.stem}_pos.txt"/>
    </logfile>
  </Output>
</febio_spec>
"""
    path.write_text(xml)
    return path


def write_elastin_feb(
    path, mesh, p: MaterialParams, pressure: float, coords=None,
    nodal_forces: np.ndarray | None = None,
    n_steps: int = 5, etol: float = 1e-16, rtol: float = 1e-16,
    min_residual: float = 1e-16, max_refs: int = 200, plot: bool = False,
    laugon: bool = True, augtol: float = 1e-8, maxaug: int = 30,
) -> Path:
    """Elastin-only model used inside the AIM (FEBio's built-in Mooney-Rivlin).

    ``nodal_forces`` optionally adds the (n_node, 3) force vector that
    represents the embedded fibre stress exactly, see
    ``kinematics.fiber_nodal_forces``.  The solver is still a black box: it
    only ever sees an isotropic material, a pressure and nodal loads.
    """
    path = Path(path)
    control = (
        "    <analysis>STATIC</analysis>\n"
        f"    <time_steps>{n_steps}</time_steps>\n"
        f"    <step_size>{1.0 / n_steps}</step_size>\n"
        "    <plot_level>PLOT_MAJOR_ITRS</plot_level>\n"
        "    <output_level>OUTPUT_FINAL</output_level>\n"
        + _SOLVER.format(etol=etol, rtol=rtol, min_residual=min_residual,
                         max_refs=max_refs)
    )
    material = (
        '    <material id="1" name="wall_material" type="Mooney-Rivlin">\n'
        "      <density>1.0</density>\n"
        f"      <c1>{p.phi_e * p.c_e:.12g}</c1>\n"
        "      <c2>0</c2>\n"
        f"      <k>{p.K:.12g}</k>\n"
        "    </material>\n"
    )
    nodesets, loads = "", ""
    if nodal_forces is not None:
        nodesets = (
            '    <NodeSet name="all">\n'
            + "      " + ",".join(str(i + 1) for i in range(mesh.n_node))
            + "\n    </NodeSet>\n"
        )
        loads = (
            '    <nodal_load name="fiber_forces" type="nodal_force" '
            'node_set="all">\n'
            '      <value type="map">fiber_force</value>\n'
            "    </nodal_load>\n"
        )
    meshdata = _fmt_node_map("fiber_force", nodal_forces) if nodal_forces is not None else ""
    return _assemble(
        path, mesh, mesh.nodes if coords is None else coords, material, pressure,
        control, "        <point>0,0</point>\n        <point>1,1</point>",
        meshdata=meshdata, domain_opts=_fmt_domain_opts(laugon, augtol, maxaug),
        extra_nodesets=nodesets, extra_loads=loads,
        plot=plot, plot_vars='      <var type="displacement"/>\n'
                             '      <var type="stress"/>\n',
    )


def write_cmm_feb(
    path, mesh, p: MaterialParams, pressure: float, coords,
    lam_ref: np.ndarray | None = None,
    J_ref: np.ndarray | None = None,
    phi_c: float = 0.0,
    k_d: float = 0.0,
    K_sigma: float = 1.0,
    t_gr: float = 1.0,
    n_steps: int = 5,
    t_end: float = 1.0,
    gr_step: float | None = None,
    etol: float = 1e-16, rtol: float = 1e-16, min_residual: float = 1e-16,
    max_refs: int = 200, plot: bool = False, adaptive: bool = True,
    laugon: bool = True, augtol: float = 1e-8, maxaug: int = 30,
) -> Path:
    """Constrained mixture model (plugin), optionally with growth & remodelling.

    ``lam_ref`` and ``J_ref`` are (n_elem, 8) arrays of ``|G_e M|`` and
    ``det(G_e)`` at the integration points; they map the stress-free elastin
    mesh ``coords`` back onto the in vivo configuration.  The pressure is ramped
    over ``t in [0, t_gr]`` in ``n_steps`` and then held constant until
    ``t_end`` in steps of ``gr_step``.
    """
    path = Path(path)
    if laugon and k_d > 0.0:
        raise ValueError(
            "laugon drives J -> 1, which fights the growth constraint J -> J_g; "
            "pass laugon=False for a run with k_d > 0"
        )
    dt = t_gr / n_steps if gr_step is None else gr_step
    n_total = int(round(max(t_gr, t_end) / dt))

    stepper = (
        '    <time_stepper type="default">\n'
        f"      <dtmin>{dt * 1e-3:.12g}</dtmin>\n"
        f"      <dtmax>{dt:.12g}</dtmax>\n"
        "      <max_retries>6</max_retries>\n"
        "      <opt_iter>8</opt_iter>\n"
        "      <aggressiveness>0</aggressiveness>\n"
        "    </time_stepper>\n"
        if adaptive else ""
    )
    control = (
        "    <analysis>STATIC</analysis>\n"
        f"    <time_steps>{n_total}</time_steps>\n"
        f"    <step_size>{dt:.12g}</step_size>\n"
        "    <plot_level>PLOT_MAJOR_ITRS</plot_level>\n"
        "    <output_level>OUTPUT_MAJOR_ITRS</output_level>\n"
        + stepper
        + _SOLVER.format(etol=etol, rtol=rtol, min_residual=min_residual,
                         max_refs=max_refs)
    )
    material = (
        '    <material id="1" name="wall_material" type="prestress_cmm">\n'
        "      <density>1.0</density>\n"
        f"      <c_e>{p.c_e:.12g}</c_e>\n"
        f"      <k>{p.K:.12g}</k>\n"
        f"      <phi_e>{p.phi_e:.12g}</phi_e>\n"
        f"      <c1>{p.c1:.12g}</c1>\n"
        f"      <c2>{p.c2:.12g}</c2>\n"
        f"      <lam_dep>{p.lam_dep:.12g}</lam_dep>\n"
        f"      <phi_c>{phi_c:.12g}</phi_c>\n"
        f"      <k_d>{k_d:.12g}</k_d>\n"
        f"      <K_sigma>{K_sigma:.12g}</K_sigma>\n"
        f"      <t_gr>{t_gr:.12g}</t_gr>\n"
        + ('      <lam_ref type="map">lam_ref</lam_ref>\n' if lam_ref is not None else "")
        + ('      <J_ref type="map">J_ref</J_ref>\n' if J_ref is not None else "")
        + '      <fiber type="cylindrical">\n'
          "        <center>0,0,0</center>\n"
          "        <axis>0,0,1</axis>\n"
          "        <vector>0,1,0</vector>\n"
          "      </fiber>\n"
        "    </material>\n"
    )
    blocks = []
    if lam_ref is not None:
        blocks.append(_fmt_mp_map("lam_ref", lam_ref))
    if J_ref is not None:
        blocks.append(_fmt_mp_map("J_ref", J_ref))

    pts = [f"        <point>0,0</point>", f"        <point>{t_gr:.12g},1</point>"]
    if t_end > t_gr:
        pts.append(f"        <point>{t_end:.12g},1</point>")
    return _assemble(
        path, mesh, coords, material, pressure, control, "\n".join(pts),
        meshdata="\n".join(blocks),
        domain_opts=_fmt_domain_opts(laugon, augtol, maxaug), plot=plot,
        plot_vars='      <var type="displacement"/>\n'
                  '      <var type="stress"/>\n'
                  '      <var type="relative volume"/>\n'
                  '      <var type="velocity"/>\n',
    )


# ---------------------------------------------------------------------------
# run / read
# ---------------------------------------------------------------------------
def run_febio(feb, quiet: bool = True, timeout: float = 7200,
              plugin: bool = False) -> Path:
    """Run FEBio on ``feb``; raise if the solver does not terminate normally."""
    feb = Path(feb)
    cmd = [febio_executable(), "-i", feb.name, "-nosplash"]
    if plugin:
        cmd += ["-import", plugin_path()]
    if quiet:
        cmd.append("-silent")
    res = subprocess.run(cmd, cwd=feb.parent, capture_output=True, text=True,
                         timeout=timeout)
    log = feb.with_suffix(".log")
    text = log.read_text() if log.is_file() else res.stdout
    if "N O R M A L   T E R M I N A T I O N" not in text:
        tail = "\n".join((text + "\n" + res.stdout).strip().splitlines()[-30:])
        raise RuntimeError(f"FEBio failed for {feb}:\n{tail}")
    return log


_STEP_RE = re.compile(r"^\*Step\s*=\s*(\d+)", re.M)


def read_node_log(path, n_node: int, n_col: int = 3, step: int = -1) -> np.ndarray:
    """Read one data block of a FEBio ``node_data`` log file -> (n_node, n_col)."""
    text = Path(path).read_text()
    blocks = _STEP_RE.split(text)[1:]          # [idx0, body0, idx1, body1, ...]
    bodies = blocks[1::2]
    body = bodies[step]
    data = np.zeros((n_node, n_col))
    seen = 0
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != n_col + 1 or line.startswith("*"):
            continue
        data[int(parts[0]) - 1] = [float(v) for v in parts[1:]]
        seen += 1
    if seen != n_node:
        raise RuntimeError(f"expected {n_node} nodes in {path}, found {seen}")
    return data


_TIME_RE = re.compile(r"^\*Time\s*=\s*(\S+)", re.M)


def read_node_log_all(path, n_node: int, n_col: int = 3):
    """Read every data block -> (n_step, n_node, n_col) and the solution times."""
    text = Path(path).read_text()
    parts = _STEP_RE.split(text)[1:]
    out, times = [], []
    for body in parts[1::2]:
        data = np.zeros((n_node, n_col))
        for line in body.splitlines():
            q = line.split()
            if len(q) != n_col + 1 or line.startswith("*"):
                continue
            data[int(q[0]) - 1] = [float(v) for v in q[1:]]
        out.append(data)
        m = _TIME_RE.search(body)
        times.append(float(m.group(1)) if m else np.nan)
    return np.array(out), np.array(times)
