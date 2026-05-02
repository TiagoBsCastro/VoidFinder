# Revised Approximate Paired-Halo Void Finder

## Purpose

This algorithm is a **fast approximate void finder** designed for **paired simulations**. It avoids expensive density-field watershed methods and instead works **directly on halo catalogs**.

The central idea is:

- compact overdense halo structures in catalog **A** act as counterparts of underdense regions in simulation **B**;
- compact overdense halo structures in catalog **B** act as counterparts of underdense regions in simulation **A**.

The method therefore proceeds in two stages:

1. **cluster halos** to identify **source clusters** and map them into **protovoids** in the paired catalog;
2. **merge protovoids** using a calibrated graph-based procedure.

The method is intended to be **calibrated against a more expensive reference void finder**.

---

## Inputs

For each paired realization, the algorithm requires:

- halo catalog **A** with halo positions and masses;
- halo catalog **B** with halo positions and masses;
- simulation box size `L`;
- periodic boundary conditions;
- mean matter density `\bar\rho_m` for the Lagrangian radius mapping;
- optional reference void catalogs used for calibration.

---

## Outputs

The algorithm returns:

- a predicted void catalog for **A**;
- a predicted void catalog for **B**;
- intermediate **source-cluster catalogs**;
- intermediate **protovoid catalogs**;
- merge graph diagnostics;
- optional evaluation statistics relative to a reference finder.

---

## Algorithm overview

### Step 1. Cluster halos in the source catalog

For each source catalog separately:

- cluster halos in **A** to define compact source clusters `C_i^(A)`;
- cluster halos in **B** to define compact source clusters `C_i^(B)`.

The clustering stage should identify **compact and robust overdense structures**, not large overmerged complexes.

A density-based method is preferred, for example:

- periodic FoF-like clustering;
- DBSCAN or HDBSCAN with periodic wrapping handled correctly.

For each source cluster, compute:

- total mass
  \[
  M_i = \sum_{h \in C_i} m_h ;
  \]
- cluster center `x_i`, preferably mass-weighted;
- number of member halos;
- effective source radius from member positions;
- shape tensor or inertia tensor;
- compactness metrics, such as concentration proxy or average radius.

These cluster properties define the seeds for the protovoid construction.

---

### Step 2. Map source clusters to protovoids in the paired catalog

Each source cluster in **A** defines one **protovoid candidate in B**.
Each source cluster in **B** defines one **protovoid candidate in A**.

The simplest first implementation uses spherical protovoids.

#### Center

The protovoid center is initially taken to be the source-cluster center:

\[
\mathbf{x}_{\rm proto} = \mathbf{x}_{\rm cluster}.
\]

#### Radius

The protovoid radius is not fixed to the exact Lagrangian radius. Instead, use a calibrated power-law relation:

\[
R_{\rm proto} = a_0\,R_L(M_i)^{\alpha},
\]

with

\[
R_L(M_i) = \left(\frac{3M_i}{4\pi\bar\rho_m}\right)^{1/3}.
\]

Here:

- `a0` is a normalization parameter;
- `alpha` is a slope parameter.

These parameters should be calibrated to reproduce the target void catalog.

#### Optional shape extension

An optional future extension is to assign ellipsoidal shapes using the source-cluster inertia tensor. The first implementation should remain spherical for simplicity.

---

### Step 3. Build the protovoid adjacency graph

For a given target catalog, the protovoids are represented as nodes of a graph.

Two protovoids `i` and `j` are connected if they are sufficiently close:

\[
d_{ij} < f_{\rm adj}(R_i + R_j),
\]

where:

- `d_ij` is the periodic distance between centers;
- `R_i`, `R_j` are the protovoid radii;
- `f_adj` is a configurable adjacency factor.

This graph defines the candidate merge pairs.

---

### Step 4. Compute a merge score for adjacent protovoid pairs

Merging should not be decided from position alone.

For each adjacent pair `(i, j)`, compute a merge score

\[
S_{ij} = w_{\rm geom} G_{ij} + w_{\rm bridge} B_{ij} + w_{\rm comp} Q_{ij},
\]

where:

- `G_ij` is a geometric proximity or overlap term;
- `B_ij` is a bridge-strength term measured in the **source halo catalog**;
- `Q_ij` is a size or quality compatibility term.

#### 4.1 Geometric term

This term favors nearby and overlapping protovoids.

Possible definitions include:

- normalized center distance;
- overlap fraction of spherical volumes;
- a soft overlap proxy based on `d_ij / (R_i + R_j)`.

#### 4.2 Bridge-strength term

This term is physically important.

Suppose the target protovoids are in **B**. Their source clusters lie in **A**. To evaluate whether the two protovoids should merge, inspect whether the two source clusters in **A** are connected by an overdense bridge.

A simple implementation is:

1. define a cylindrical or capsule-shaped region joining the two source-cluster centers;
2. measure the halo number density and/or halo mass density inside this region;
3. normalize by the mean halo density or by a local background estimate.

A stronger bridge favors merging.

#### 4.3 Compatibility term

This term encodes whether the two protovoids are similar enough to be part of the same larger structure.

Possible ingredients include:

- ratio of radii;
- similarity of source-cluster compactness;
- similarity in source-cluster richness;
- optional shape compatibility.

---

### Step 5. Merge protovoids into final voids

Two protovoids are merged if their score exceeds a threshold:

\[
S_{ij} > S_{\rm merge}.
\]

After thresholding the graph, the remaining connected components define the final merged voids.

For each final void, compute:

- final center;
- effective radius;
- member protovoids;
- associated source clusters;
- total associated source-cluster mass;
- optional shape quantities.

#### Radius of a merged void

At least two options should be supported:

1. **volume-sum radius**:
   sum the spherical member volumes and convert the total volume into an effective radius;
2. **mass-sum radius**:
   sum the associated source-cluster masses and apply the calibrated mass-to-radius mapping again.

The choice should be configurable and tested against the reference void finder.

---

## Symmetry requirement

The method must be symmetric under swapping `A` and `B`.

That is:

- clusters in **A** define voids in **B**;
- clusters in **B** define voids in **A**;
- the same code path should be used in both directions.

This symmetry should be explicitly tested.

---

## Free parameters to calibrate

The main parameters are:

### Halo clustering

- `linking_length`
- `min_cluster_members`
- `min_cluster_mass`

### Cluster-to-protovoid mapping

- `radius_a0`
- `radius_alpha`
- `reference_rho_bar`

### Protovoid adjacency

- `adjacency_factor`

### Bridge metric

- `bridge_radius_factor`
- optional bridge-density normalization choices

### Merge score

- `geom_weight`
- `bridge_weight`
- `compatibility_weight`
- `merge_threshold`

### Final catalog cuts

- `merged_radius_mode`
- `min_void_radius`

---

## Recommended calibration strategy

The method is intended to be calibrated to reproduce a more expensive reference void finder.

A sensible calibration order is:

### First stage: seed and size calibration

Fit:

- `linking_length`
- `radius_a0`
- `radius_alpha`

using the agreement with the reference **void size function**.

### Second stage: merge calibration

Fit:

- `adjacency_factor`
- `bridge_radius_factor`
- `merge_threshold`

in order to control overmerging and fragmentation.

### Third stage: score refinement

Adjust:

- `geom_weight`
- `bridge_weight`
- `compatibility_weight`

if additional refinement is needed.

---

## Suggested calibration targets

The objective function may combine several diagnostics:

- agreement of the void abundance or size function;
- matching of void centers;
- radius agreement;
- overlap score between predicted and reference voids;
- optional agreement in stacked radial profiles.

Calibration and evaluation should be performed on separate paired realizations.

---

## Recommended development order

### Phase 1. Minimal working prototype

Implement:

- periodic FoF-like halo clustering;
- spherical protovoids;
- adjacency graph from center distance;
- merge score from geometric proximity, bridge density, and compatibility;
- connected-component merging;
- symmetric A-to-B and B-to-A execution.

### Phase 2. Calibration and evaluation

Implement:

- parameter sweeps or an optimizer;
- comparison to reference void catalogs;
- diagnostic plots and benchmark statistics.

### Phase 3. Extensions

Possible extensions include:

- DBSCAN or HDBSCAN clustering backend;
- ellipsoidal protovoids;
- more refined bridge metrics;
- learned merge scores.

---

## Compact pseudocode

```python
for source, target in [(A, "B"), (B, "A")]:
    clusters = cluster_halos(source)
    cluster_props = summarize_source_clusters(clusters)
    protovoids = source_clusters_to_protovoids(cluster_props)
    graph = build_protovoid_graph(protovoids)

    for edge (i, j) in graph:
        geom = compute_geometric_term(protovoids[i], protovoids[j])
        bridge = compute_bridge_metric(source, clusters[i], clusters[j])
        compat = compute_compatibility_term(protovoids[i], protovoids[j])
        score = compute_merge_score(geom, bridge, compat)
        keep edge if score > threshold

    final_voids[target] = merge_protovoids(graph, protovoids)
```

---

## Why this method is useful

This method is attractive because it:

- works directly on halo catalogs;
- avoids expensive watershed segmentation;
- separates **seed identification** from **void merging**;
- is modular and easy to calibrate;
- preserves the physical symmetry of paired simulations;
- is naturally suited for fast approximate void catalogs.

---

## Summary

The revised approximate paired-halo void finder is defined by the pipeline:

**halo clustering -> source clusters -> protovoids -> protovoid graph -> calibrated merging -> final void catalog**

Its main novelty is that it uses **paired halo catalogs** to construct voids through a fast, calibratable surrogate model, rather than through a full density-field watershed method.
