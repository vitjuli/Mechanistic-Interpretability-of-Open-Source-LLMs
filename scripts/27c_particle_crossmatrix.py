"""
27c_particle_crossmatrix.py  —  particle clusters x geometry, WITH cross-effects
================================================================================
Particle-cluster version of 27c, extended with Iuliia's cross-matrix idea: ablate each
co-importance cluster and measure its effect not only on ITS OWN axis but on OTHER axes,
to test specificity vs leakage. Measures BOTH length and rotation (Iuliia: a cluster may
ADD to its own delta but ROTATE a neighbouring one -- both matter).

Clusters: cluster_membership_particles.csv (8 co-importance clusters, enriched photon/neutron).
Corpus: neutron_vs_photon pair-slice (delta = photon - neutron = exactly the axis clusters encode).
  HONEST CAVEAT (printed): clusters defined on full multiclass (correct-vs-incorrect); delta on the
  photon/neutron pair. Same feature space, slightly different contrast. Defensible; note in thesis.

Ablation: DIRECT SUBTRACTION (carrier_geom_core), no transcoder-error injection. carrier_geom_core
already decomposes each axis change into signed LENGTH + ROTATION (both reported).

CROSS-MATRIX (the new part):
  rows  = 8 clusters (each enriched for a particle)
  cols  = axes: delta (photon-neutron) [+ optionally u per particle if grad dumps available]
  cell  = {length_change, rotation_mag} of that axis when this cluster is ablated, vs same-size null
  Built-in control: a photon-enriched cluster should move the photon side; effect on a pair NOT
  containing its particle is an implicit control (expect ~null).

RUN (CSD3):
  python 27c_particle_crossmatrix.py \
    --membership cluster_membership_particles.csv --pool particle_pool_by_layer.json \
    --prompts data/prompts/particle_pairs/particles_neutron_vs_photon.jsonl \
    --label_key _pair_class --pos_value 1 --want_u --n_null 20 \
    --out particle_crossmatrix.csv
"""
from __future__ import annotations
import argparse, json, logging
import numpy as np
import carrier_geom_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pcross")

def load_membership_particles(path):
    """cluster_membership_particles.csv -> {cluster: {layer:[feat]}} + labels + enriched particle."""
    import csv
    groups={}; labels={}; particle={}
    for r in csv.DictReader(open(path)):
        fid=r["feature_id"]; L=int(r["layer"]); feat=int(fid.split("_F")[1]); cid=r["cluster"]
        groups.setdefault(cid,{}).setdefault(L,[]).append(feat)
        labels[cid]=r.get("semantic_label",cid); particle[cid]=r.get("enriched_particle","?")
    return groups, labels, particle

def load_pool_json(path):
    return {int(k):v for k,v in json.load(open(path)).items()}

def run(args):
    groups_raw, labels, particle = load_membership_particles(args.membership)
    pool = load_pool_json(args.pool)
    groups = {f"C{cid}_{particle[cid]}": gmap for cid, gmap in groups_raw.items()}
    all_layers = sorted({L for gmap in groups_raw.values() for L in gmap})
    logger.info("particle clusters: %d | layers %s", len(groups), all_layers)
    for cid, gmap in groups_raw.items():
        logger.info("  C%s [%s] enriched=%s layers=%s n=%d", cid, labels[cid][:30],
                    particle[cid], sorted(gmap), sum(len(v) for v in gmap.values()))

    prompts = [json.loads(l) for l in open(args.prompts)]
    def lab(p):
        if args.label_key:
            v=p.get(args.label_key)
            return int(str(v).strip().lower()==str(args.pos_value).strip().lower())
        v=p.get("label",p.get("answer"))
        return 1 if (isinstance(v,str) and v.strip().lower().startswith(("b","p","photon"))) else 0
    y=np.array([lab(p) for p in prompts])
    logger.info("corpus %d (class0=%d class1=%d) -- delta=class1-class0=photon-neutron",
                len(y), int((y==0).sum()), int((y==1).sum()))

    model, tok, tcs = core.load_model_and_transcoders()
    results = core.run_group_geometry(model, tok, tcs, prompts, y, groups, all_layers,
                                      want_u=args.want_u, n_null=args.n_null,
                                      null_pool_by_layer=pool, seed=args.seed)
    core.write_results(results, args.out, want_u=args.want_u)
    json.dump({"labels":labels,"particle":particle}, open(args.out+".meta.json","w"), indent=2)

    # cross-matrix summary: per cluster, the LENGTH and ROTATION imprint on delta (its own axis),
    # vs null, with the enriched particle noted -> read specificity.
    print(f"\nPARTICLE CROSS-MATRIX (cluster -> delta geometry; both length & rotation):")
    print(f"{'cluster':22} {'particle':9} {'delta_len':>10} {'len_z':>7} {'delta_rot':>10} {'rot_z':>7}")
    print("-"*70)
    for res in results:
        g=res["group"]; cid=g.split("_")[0][1:]; part=particle.get(cid,"?")
        # aggregate delta geometry across layers (mean |len|, mean rot), and z vs null
        null_by=defaultdict(lambda:{"rot":[],"len":[]})
        for nd in res["null"]:
            for r in nd:
                if r["axis"]=="delta":
                    null_by["d"]["rot"].append(r["rot_mag"]); null_by["d"]["len"].append(r["len_change"])
        gl=[r["len_change"] for r in res["geometry"] if r["axis"]=="delta"]
        gr=[r["rot_mag"] for r in res["geometry"] if r["axis"]=="delta"]
        ml=float(np.mean(np.abs(gl))) if gl else float("nan")
        mr=float(np.mean(gr)) if gr else float("nan")
        nl=null_by["d"]["len"]; nr=null_by["d"]["rot"]
        lz=(np.mean(np.abs(gl))-np.mean(np.abs(nl)))/(np.std(np.abs(nl))+1e-9) if nl else float("nan")
        rz=(np.mean(gr)-np.mean(nr))/(np.std(nr)+1e-9) if nr else float("nan")
        print(f"{g:22} {part:9} {ml:>10.3f} {lz:>7.1f} {mr:>10.3f} {rz:>7.1f}")
    print(f"\nREAD: do photon-enriched clusters move delta(photon-neutron) more than neutron-enriched ones?")
    print(f"      LENGTH vs ROTATION may dissociate (a cluster may lengthen its own axis but rotate it,")
    print(f"      or move a neighbouring particle's axis -- specificity vs leakage). z vs same-size null.")

from collections import defaultdict
def build_parser():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--membership",required=True); p.add_argument("--pool",required=True)
    p.add_argument("--prompts",required=True); p.add_argument("--label_key",default=None)
    p.add_argument("--pos_value",default=None); p.add_argument("--want_u",action="store_true")
    p.add_argument("--n_null",type=int,default=20); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out",default="particle_crossmatrix.csv")
    return p

if __name__=="__main__":
    run(build_parser().parse_args())
