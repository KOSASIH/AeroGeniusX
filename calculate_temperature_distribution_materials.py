import numpy as np

def calculate_temperature_distribution(material_properties, heat_transfer_mechanisms, external_heat_flux, structure_dimensions):
    """
    Calculates the temperature distribution in the aircraft's structure based on the thermal properties of the materials,
    heat transfer mechanisms, external heat flux, and structure dimensions.

    Args:
        material_properties (dict): A dictionary containing the thermal properties of the materials used in the structure.
        heat_transfer_mechanisms (dict): A dictionary specifying the heat transfer mechanisms for each material.
        external_heat_flux (float): The external heat flux applied to the structure.
        structure_dimensions (dict): A dictionary containing the dimensions of the structure.

    Returns:
        dict: A dictionary containing the temperature distribution, heat flux, and thermal protection effectiveness.

    """
    num_nodes = int(structure_dimensions['length'] / structure_dimensions['thickness'])

    temperature_distribution = np.zeros(num_nodes)
    heat_flux = np.zeros(num_nodes)
    thermal_protection_effectiveness = np.zeros(num_nodes)

    for i in range(num_nodes):
        temperature = external_heat_flux / (material_properties['conductivity'] * structure_dimensions['thickness'])
        temperature_distribution[i] = temperature

        if i == 0:
            heat_flux[i] = (temperature_distribution[i + 1] - temperature_distribution[i]) / structure_dimensions['thickness']
        elif i == num_nodes - 1:
            heat_flux[i] = (temperature_distribution[i] - temperature_distribution[i - 1]) / structure_dimensions['thickness']
        else:
            heat_flux[i] = (temperature_distribution[i + 1] - temperature_distribution[i - 1]) / (2 * structure_dimensions['thickness'])

        thermal_protection_effectiveness[i] = (external_heat_flux - heat_flux[i]) / external_heat_flux

    output = {
        'temperature_distribution': temperature_distribution,
        'heat_flux': heat_flux,
        'thermal_protection_effectiveness': thermal_protection_effectiveness
    }

    return output

def print_results(analysis_result):
    """
    Prints the analysis results in markdown format.

    Args:
        analysis_result (dict): A dictionary containing the temperature distribution, heat flux, and thermal protection effectiveness.

    """
    print("## Thermal Performance Analysis")
    print("\n### Temperature Distribution")
    print("\n| Node | Temperature (°C) |")
    print("| ---- | --------------- |")
    for i, temperature in enumerate(analysis_result['temperature_distribution']):
        print(f"| {i+1} | {temperature:.2f} |")

    print("\n### Heat Flux")
    print("\n| Node | Heat Flux (W/m^2) |")
    print("| ---- | ---------------- |")
    for i, flux in enumerate(analysis_result['heat_flux']):
        print(f"| {i+1} | {flux:.2f} |")

    print("\n### Thermal Protection Effectiveness")
    print("\n| Node | Effectiveness |")
    print("| ---- | ------------ |")
    for i, effectiveness in enumerate(analysis_result['thermal_protection_effectiveness']):
        print(f"| {i+1} | {effectiveness:.2f} |")

# Example usage
material_properties = {
    'conductivity': 0.5  # Thermal conductivity of the material in W/(m*K)
}

heat_transfer_mechanisms = {
    'material': 'conduction'  # Heat transfer mechanism for the material
}

external_heat_flux = 100  # External heat flux applied to the structure in W/m^2

structure_dimensions = {
    'length': 10,  # Length of the structure in meters
    'width': 5,  # Width of the structure in meters
    'thickness': 0.1  # Thickness of the structure in meters
}

analysis_result = calculate_temperature_distribution(material_properties, heat_transfer_mechanisms, external_heat_flux, structure_dimensions)
print_results(analysis_result)
