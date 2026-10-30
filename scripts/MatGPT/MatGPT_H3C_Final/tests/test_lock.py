"""Pruebas de integridad para la arquitectura H3-C fijada y su reproducción."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from Alternative2.scripts.MatGPT.MatGPT_H3C_Final.reproduce import (
    EXPECTED_CHECKPOINT,FAMILIES,read_development,sha256)
from Alternative2.scripts.MatGPT.MatGPT_H3C_Final.controls import formula_fractions,roles
from Alternative2.scripts.MatGPT.MatGPT_H3C_Final.interface import load_inputs,predict_triplets,evaluate_intraformula
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import StructuralMLP8


class H3CLockTests(unittest.TestCase):
    """Bloquea hashes, dimensiones, particiones y conversiones físicas."""

    @classmethod
    def setUpClass(cls) -> None:
        """Lee únicamente fuentes de desarrollo y resultados ya reproducidos."""
        cls.report=ROOT/"reports/MatGPT/MatGPT_H3C_Final"
        cls.lock=yaml.safe_load((ROOT/"scripts/MatGPT/MatGPT_H3C_Final/h3c_architecture_lock.yaml").read_text())
        cls.data=read_development(ROOT/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv")
        cls.replayed=pd.read_csv(cls.report/"reproduction/replayed_predictions.csv")

    def test_checkpoint_and_tokenizer_hashes(self) -> None:
        """El backbone y vocabulario originales concuerdan con el lock."""
        model=ROOT/"models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
        info=model.with_name("model_info_property_0.json")
        self.assertEqual(sha256(model),EXPECTED_CHECKPOINT)
        self.assertEqual(sha256(info),self.lock["extractor"]["tokenizer_model_info_sha256"])

    def test_feature_hash_and_shapes(self) -> None:
        """Ambas representaciones históricas tienen 210×768 y hash fijo."""
        path=ROOT/"reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
        self.assertEqual(sha256(path),self.lock["extractor"]["feature_cache_sha256"])
        with np.load(path,allow_pickle=False) as cache:
            self.assertEqual(cache["structure_id"].astype(str).tolist(),self.data.structure_id.tolist())
            self.assertEqual(cache["L03_MEAN"].shape,(210,768))
            self.assertEqual(cache["FINAL_END"].shape,(210,768))

    def test_architecture_exact(self) -> None:
        """Cambios de capas, ancho, activación o dropout invalidan el lock."""
        model=StructuralMLP8(input_dim=768,dropout=.1)
        self.assertEqual(sum(p.numel() for p in model.parameters()),6161)
        self.assertEqual([type(layer).__name__ for layer in model.network],
                         ["Linear","GELU","Dropout","Linear"])
        self.assertEqual(model.network[0].in_features,768)
        self.assertEqual(model.network[0].out_features,8)
        self.assertEqual(model.network[3].out_features,1)
        self.assertAlmostEqual(model.network[2].p,.1)
        self.assertEqual(self.lock["structural_mlp8"]["target"],
                         "DFT energy per atom minus formula mean, eV/atom")

    def test_replayed_predictions_match_historical(self) -> None:
        """Los 630 valores se reproducen al recargar checkpoints."""
        manifest=json.loads((self.report/"reproduction/reproduction_manifest.json").read_text())
        self.assertEqual(manifest["status"],"PASS")
        self.assertEqual(len(self.replayed),630)
        self.assertLess(max(manifest["max_abs_error_ev_atom"].values()),1e-5)

    def test_dataset_and_targets(self) -> None:
        """Cada ID conserva energía total y división exacta por número de átomos."""
        self.assertEqual(len(self.data),210)
        self.assertEqual(self.data.composition_id.nunique(),70)
        self.assertTrue((self.data.groupby("composition_id").size()==3).all())
        self.assertTrue(np.allclose(self.data.energy_total_dft_ev/self.data.n_atoms,
                                    self.data.energy_per_atom_dft_ev,atol=1e-8))
        self.assertTrue(np.allclose(self.replayed.energy_total_pred_ev/self.replayed.n_atoms,
                                    self.replayed.energy_per_atom_pred_ev,atol=1e-8))
        self.assertFalse(self.data.structure_id.str.contains("::test").any())

    def test_folds_and_no_leakage(self) -> None:
        """Se conservan los cinco pliegues históricos por composición."""
        folds=pd.read_csv(ROOT/"reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv")
        folds=folds[folds.historical_split.isin(["train","val"])]
        for fold in range(5):
            tr,va,outer,fit=roles(self.data,folds,fold)
            self.assertEqual((len(tr),len(va),len(outer),len(fit)),(135,33,42,168))

    def test_energy_differences(self) -> None:
        """ΔE es antisimétrico y transitivo tras calcular E/N."""
        group=self.replayed[self.replayed.composition_id==self.replayed.composition_id.iloc[0]]
        group=group[group.seed==42].set_index("polymorph_family").loc[list(FAMILIES)]
        e=group.energy_per_atom_pred_ev.to_numpy(float)
        d01=1000*(e[1]-e[0]);d10=1000*(e[0]-e[1])
        d12=1000*(e[2]-e[1]);d02=1000*(e[2]-e[0])
        self.assertAlmostEqual(d01,-d10)
        self.assertAlmostEqual(d01+d12,d02)
        self.assertEqual(int(np.argmin(e)),int(np.argmin(group.energy_total_pred_ev/group.n_atoms)))

    def test_composition_identity_and_ties(self) -> None:
        """El control de fórmula no recibe pistas polimórficas ni adjudica empates."""
        fractions,elements=formula_fractions(self.data.composition_id.tolist())
        self.assertGreater(len(elements),2)
        for _,part in self.data.reset_index().groupby("composition_id"):
            self.assertTrue(np.array_equal(fractions[part["index"].iloc[0]],fractions[part["index"].iloc[1]]))
        pred=pd.read_csv(self.report/"controls/composition_seed_42/predictions.csv")
        self.assertTrue(pred.groupby("composition_id").energy_per_atom_pred_ev.nunique().eq(1).all())
        summary=pd.read_csv(self.report/"analysis/summary_by_seed.csv")
        self.assertEqual(int(summary[summary.model=="composition"].gs_correct.sum()),0)

    def test_partner_context_is_not_independent(self) -> None:
        """El centrado residual depende del triplete; no se etiqueta Siamese independiente."""
        torch.manual_seed(42)
        model=StructuralMLP8().eval()
        a=torch.randn(1,3,768)
        b=a.clone();b[0,1]+=1
        with torch.inference_mode():
            x=model(a)[0,0].item();y=model(b)[0,0].item()
        self.assertNotAlmostEqual(x,y,places=7)

    def test_random_control_hashes(self) -> None:
        """Tres inicializaciones congeladas poseen hashes distintos del checkpoint."""
        lock=json.loads((ROOT/"scripts/MatGPT/MatGPT_H3C_Final/random_initialization_lock.json").read_text())
        self.assertEqual(len(set(lock["state_tensor_sha256"].values())),3)
        for seed in (42,123,2026):
            m=json.loads((self.report/f"controls/random_seed_{seed}/manifest.json").read_text())
            self.assertEqual(m["status"],"PASS")
            self.assertEqual(m["backbone_trainable_parameters"],0)
            self.assertFalse(m["historical_test_accessed"])

    def test_stable_interface_matches_saved_prediction(self) -> None:
        """La interfaz futura conserva centrado triplete y protege DFT por defecto."""
        bundle=load_inputs(ROOT)
        self.assertNotIn("energy_per_atom_dft_ev",bundle.metadata)
        expected=self.replayed[(self.replayed.fold==0)&(self.replayed.seed==42)]
        actual=predict_triplets(ROOT,0,42,expected.structure_id.tolist(),bundle)
        aligned=expected[["structure_id","energy_per_atom_pred_ev"]].merge(
            actual[["structure_id","energy_per_atom_pred_ev"]],on="structure_id",validate="one_to_one",suffixes=("_saved","_interface"))
        self.assertEqual(len(aligned),42)
        self.assertLess(np.max(np.abs(aligned.energy_per_atom_pred_ev_saved-aligned.energy_per_atom_pred_ev_interface)),1e-5)
        with_dft=load_inputs(ROOT,include_dft=True)
        evaluated=evaluate_intraformula(actual,with_dft.metadata)
        self.assertEqual(len(evaluated),42)


if __name__=="__main__":unittest.main()
