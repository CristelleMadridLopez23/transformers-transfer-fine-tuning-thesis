"""Pruebas de integridad y estimandos de la auditoría estadística offline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODULE_DIR))
import audit_statistics as audit


class AnchoredEnergyTests(unittest.TestCase):
    """Comprueba la referencia DFT común y las unidades de los gaps."""

    def test_anchor_is_same_dft_selected_structure_and_zero(self) -> None:
        """El ancla DFT debe dar gaps cero en ambos métodos."""
        dft=np.array([-5.0,-4.99,-4.97]); pred=np.array([-4.7,-4.72,-4.68])
        dg,pg,anchor=audit.calculate_anchored_gaps(dft,pred)
        self.assertEqual(anchor,0); self.assertEqual(dg[anchor],0.0); self.assertEqual(pg[anchor],0.0)
        np.testing.assert_allclose(dg,[0,10,30],atol=1e-10)

    def test_common_energy_shift_does_not_change_anchored_gaps(self) -> None:
        """Un offset común por fórmula se cancela en las diferencias ancladas."""
        dft=np.array([-5.0,-4.99,-4.97]); pred=np.array([-4.7,-4.72,-4.68])
        dg,pg,_=audit.calculate_anchored_gaps(dft,pred)
        dg_shift,pg_shift,_=audit.calculate_anchored_gaps(dft,pred+7.3)
        np.testing.assert_allclose(dg,dg_shift,atol=1e-10)
        np.testing.assert_allclose(pg,pg_shift,atol=1e-10)

    def test_invalid_triplet_is_rejected(self) -> None:
        """La función no acepta tripletes incompletos o valores no finitos."""
        with self.assertRaises(ValueError): audit.calculate_anchored_gaps(np.array([-1.,-.9]),np.array([-1.,-.8]))


class SeedEstimandTests(unittest.TestCase):
    """Distingue promedio de corridas individuales del predictor ensemble."""

    def test_seed_average_mae_differs_from_mae_of_ensemble(self) -> None:
        """Errores con signo opuesto se cancelan solo al promediar predicciones."""
        rows=[]; truth=np.array([-5.0,-4.99,-4.98])
        for seed,offset in ((42,.01),(123,-.01)):
            for family,e in zip(("dist_perovskite","hexagonal","needle"),truth):
                pred=e+offset; n=20
                rows.append({"condition":"F2","fold":0,"seed":seed,"structure_id":f"S_{family}",
                    "composition_id":"A1S3","polymorph_family":family,"n_atoms":n,
                    "energy_total_dft_ev":e*n,"energy_per_atom_dft_ev":e,"ridge_mean_ev_atom":-4.99,
                    "energy_per_atom_pred_ev":pred,"energy_total_pred_ev":pred*n})
        frame=pd.DataFrame(rows); individual=audit.formula_seed_metrics(frame)
        ens=audit.ensemble_predictions(frame).assign(seed=-1); ensemble=audit.formula_seed_metrics(ens)
        self.assertAlmostEqual(individual.mae_energy_mev_atom.mean(),10.0,places=8)
        self.assertAlmostEqual(ensemble.mae_energy_mev_atom.mean(),0.0,places=8)

    def test_signed_pair_is_antisymmetric(self) -> None:
        """Invertir A y B invierte ΔE sin cambiar su magnitud."""
        ea,eb=-5.0,-4.99; ab=(eb-ea)*1000; ba=(ea-eb)*1000
        self.assertAlmostEqual(ab,-ba); self.assertAlmostEqual(abs(ab),10.0)


class ExistingDataIntegrityTests(unittest.TestCase):
    """Audita los archivos existentes sin usar targets del test histórico."""

    def test_oof_sources_are_paired_and_hash_aligned(self) -> None:
        """Las predicciones físicas, pares, pliegues y fuente F0 superan integridad."""
        dev=audit.load_development_dataset(audit.DATASET)
        pred=pd.read_csv(audit.EXPERIMENT/"predicciones_oof_por_estructura.csv")
        pairs=pd.read_csv(audit.EXPERIMENT/"predicciones_oof_por_par.csv")
        folds=pd.read_csv(audit.FOLDS); frozen=pd.read_csv(audit.F0_SOURCE)
        summary=__import__("json").loads((audit.EXPERIMENT/"summary.json").read_text())
        result=audit.validate_integrity(pred,pairs,dev,folds,frozen,summary)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["n_structures"],210); self.assertEqual(result["n_compositions"],70)
        self.assertEqual(result["n_eligible_physical_pairs"],197)
        self.assertFalse(result["test_labels_read"])


if __name__=="__main__":
    unittest.main()
