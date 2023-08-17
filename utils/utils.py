import numpy as np
import matminer.featurizers.composition.composite as composite
import pymatgen.core as mg

# Magpie Preset
magpie_preset = composite.ElementProperty.from_preset("magpie")


def get_formula_to_feature(formula: str) -> list:
    try:
        mg_comp = mg.Composition(formula)
        return magpie_preset.featurize(mg_comp)

    except Exception as e:
        # print(f"Error: {e}")
        return []


def identify_material_and_dopants(formula: str) -> dict:
    composition = mg.Composition(formula).fractional_composition

    base_material = {el: amt for el, amt in composition.items() if amt >= 0.1}
    dopants = {el: amt for el, amt in composition.items() if amt < 0.1}

    return {"base_material": base_material, "dopants": dopants}


def get_feature(formula: str) -> list:
    try:
        comp_dict = identify_material_and_dopants(formula)
        feature = []

        base_mat_str = ""
        for el, frac in comp_dict["base_material"].items():
            base_mat_str += el.symbol + str(frac)

        feature = get_formula_to_feature(base_mat_str)

        if len(comp_dict["dopants"]) > 0:
            dopants_str = ""
            sum_frac = 0
            for el, frac in comp_dict["dopants"].items():
                dopants_str += el.symbol + str(frac)
                sum_frac += frac
            dopants_feature = sum_frac * np.array(get_formula_to_feature(dopants_str))
        else:
            dopants_feature = [0] * len(feature)

        feature.extend(list(dopants_feature))

        return feature

    except Exception as e:
        # print(f"Error: {e}")
        return []
