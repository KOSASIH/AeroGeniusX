import numpy as np

def calculate_temperature_distribution(material_properties, heat_transfer, external_heat_flux, num_nodes=100, length=10.0):
    """
    Calculates the temperature distribution in an aircraft's thermal protection system.

    Args:
        material_properties (dict): Dictionary containing the thermal properties of the materials used in the system.
        heat_transfer (dict): Dictionary containing the heat transfer mechanisms and their corresponding coefficients.
        external_heat_flux (float): External heat flux applied to the system.
        num_nodes (int): Number of nodes to discretize the structure.
        length (float): Length of the structure in meters.

    Returns:
        str: Markdown code presenting a comprehensive analysis of the temperature distribution, heat flux,
             and thermal protection effectiveness in the aircraft's structure.
    """
    delta_x = length / num_nodes
    temperature = np.zeros(num_nodes)

    for i in range(1, num_nodes - 1):
        heat_flux = calculate_heat_flux(temperature[i-1], temperature[i+1], delta_x, heat_transfer)
        temperature[i] = calculate_temperature(temperature[i], heat_flux, material_properties)

    average_temperature = np.mean(temperature)
    thermal_protection_effectiveness = calculate_thermal_protection_effectiveness(average_temperature, external_heat_flux)

    output = f"## Thermal Performance Analysis\n\n"
    output += f"### Temperature Distribution\n\n"
    output += f"| Node | Temperature (°C) |\n"
    output += f"| ---- | --------------- |\n"
    for i in range(num_nodes):
        output += f"| {i+1} | {temperature[i]:.2f} |\n"
    output += f"\n"
    output += f"### Heat Flux Analysis\n\n"
    output += f"| Node | Heat Flux (W/m^2) |\n"
    output += f"| ---- | ---------------- |\n"
    for i in range(num_nodes):
        heat_flux = calculate_heat_flux(temperature[i-1], temperature[i+1], delta_x, heat_transfer)
        output += f"| {i+1} | {heat_flux:.2f} |\n"
    output += f"\n"
    output += f"### Thermal Protection Effectiveness\n\n"
    output += f"The average temperature in the structure is {average_temperature:.2f} °C.\n"
    output += f"The thermal protection effectiveness is {thermal_protection_effectiveness:.2f}%."

    return output

def calculate_heat_flux(temperature_prev, temperature_next, delta_x, heat_transfer):
    conduction_coefficient = heat_transfer['conduction']
    convection_coefficient = heat_transfer['convection']

    heat_flux = (conduction_coefficient * (temperature_prev - temperature_next) / delta_x) + convection_coefficient

    return heat_flux

def calculate_temperature(temperature, heat_flux, material_properties):
    thermal_conductivity = material_properties['thermal_conductivity']

    temperature_change = heat_flux / thermal_conductivity

    return temperature + temperature_change

def calculate_thermal_protection_effectiveness(average_temperature, external_heat_flux):
    thermal_protection_effectiveness = (1 - (average_temperature / external_heat_flux)) * 100

    return thermal_protection_effectiveness

material_properties = {
    'thermal_conductivity': 0.5
}

heat_transfer = {
    'conduction': 0.1,
    'convection': 5.0
}

external_heat_flux = 100.0

output = calculate_temperature_distribution(material_properties, heat_transfer, external_heat_flux)
print(output)
