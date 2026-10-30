import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from core import ResidualMLP8,matched_pair,seeded,centered_loss,triplets,load_geometry,GEOM_COLS
from evaluar import metrics,bootstrap

class TestGeometryFusion(unittest.TestCase):
    def test_locked_dimensions(self):
        self.assertEqual(sum(p.numel() for p in ResidualMLP8(768).parameters()),6161)
        self.assertEqual(sum(p.numel() for p in ResidualMLP8(775).parameters()),6217)
    def test_exact_same_initial_predictions(self):
        seeded(42)
        b,f=matched_pair(768,7)
        b.eval();f.eval()
        x=torch.randn(4,3,768);g=torch.randn(4,3,7)
        with torch.inference_mode():self.assertTrue(torch.allclose(b(x),f(torch.cat([x,g],dim=-1)),atol=1e-6,rtol=1e-5))
    def test_centered_and_permutation_equivariant(self):
        seeded(42)
        m=ResidualMLP8(775).eval()
        x=torch.randn(2,3,775)
        with torch.inference_mode():
            v=m(x);w=m(x[:,[2,0,1],:])
        self.assertTrue(torch.allclose(v.sum(dim=1),torch.zeros(2),atol=1e-6))
        self.assertTrue(torch.allclose(w,v[:,[2,0,1]],atol=1e-7))
    def test_geometry_gradient(self):
        seeded(42);_,model=matched_pair(768,7)
        x=torch.randn(4,3,775);y=torch.randn(4,3)
        loss=centered_loss(model(x),y-y.mean(dim=1,keepdim=True),0.1)
        loss.backward()
        self.assertGreater(float(model.network[0].weight.grad[:,768:].abs().sum()),0)
    def test_ridge_cancels_from_deltas(self):
        ridge=-5.2;res=np.array([-.03,.005,.025]);energy=ridge+res
        self.assertAlmostEqual(1000*(energy[1]-energy[0]),1000*(res[1]-res[0]))
    def test_metric_truth(self):
        rows=[]
        for cond in ['slices_only','slices_plus_cell_geometry']:
            for seed in [42,123,2026]:
                for n in range(70):
                    y=np.array([-5.,-4.98,-4.90]);pred=y.copy() if cond!='slices_only' else np.full(3,-4.96)
                    for i,fam in enumerate(('dist_perovskite','hexagonal','needle')):
                        rows.append(dict(condition=cond,seed=seed,composition_id=f'F{n}',structure_id=f'{n}_{fam}',
                                         polymorph_family=fam,n_atoms=20,fold=n%5,
                                         energy_per_atom_dft_ev=y[i],energy_total_dft_ev=20*y[i],
                                         energy_per_atom_pred_ev=pred[i],energy_total_pred_ev=20*pred[i]))
        summary,formula,pairs,bins=metrics(pd.DataFrame(rows))
        self.assertEqual(len(pairs),1260)
        self.assertEqual(len(formula),420)
        self.assertTrue(np.all(summary[summary.condition=='slices_plus_cell_geometry'].mae_delta_pairs_mev==0))
        out=bootstrap(formula,n_reps=50)
        self.assertIn('pairwise_accuracy_gt5',out.metric.tolist())

if __name__=='__main__':unittest.main()
