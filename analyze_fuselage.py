import numpy as np

def analyze_fuselage(fuselage_geometry, material_properties, applied_loads):
    # Extract fuselage geometry parameters
    length = fuselage_geometry['length']
    diameter = fuselage_geometry['diameter']
    thickness = fuselage_geometry['thickness']

    # Extract material properties
    modulus_of_elasticity = material_properties['modulus_of_elasticity']
    yield_strength = material_properties['yield_strength']

    # Extract applied loads
    axial_load = applied_loads['axial_load']
    bending_moment = applied_loads['bending_moment']

    # Calculate cross-sectional area and second moment of area
    area = np.pi * (diameter / 2) ** 2
    moment_of_area = np.pi * (diameter / 2) ** 4 / 4

    # Calculate axial stress
    axial_stress = axial_load / area

    # Calculate bending stress
    bending_stress = bending_moment * (diameter / 2) / moment_of_area

    # Calculate maximum stress
    maximum_stress = max(abs(axial_stress), abs(bending_stress))

    # Calculate safety factor
    safety_factor = yield_strength / maximum_stress

    # Calculate deformation
    deformation = bending_moment * length / (modulus_of_elasticity * moment_of_area)

    # Output the analysis results
    print("Fuselage Structural Analysis Results:")
    print("------------------------------------")
    print(f"Length: {length} m")
    print(f"Diameter: {diameter} m")
    print(f"Thickness: {thickness} m")
    print(f"Modulus of Elasticity: {modulus_of_elasticity} Pa")
    print(f"Yield Strength: {yield_strength} Pa")
    print(f"Axial Load: {axial_load} N")
    print(f"Bending Moment: {bending_moment} Nm")
    print("------------------------------------")
    print(f"Axial Stress: {axial_stress} Pa")
    print(f"Bending Stress: {bending_stress} Pa")
    print(f"Maximum Stress: {maximum_stress} Pa")
    print(f"Safety Factor: {safety_factor}")
    print(f"Deformation: {deformation} m")

# Example usage
fuselage_geometry = {
    'length': 10.0,
    'diameter': 2.0,
    'thickness': 0.05
}

material_properties = {
    'modulus_of_elasticity': 70e9,
    'yield_strength': 250e6
}

applied_loads = {
    'axial_load': 50000,
    'bending_moment': 1000000
}

analyze_fuselage(fuselage_geometry, material_properties, applied_loads)
