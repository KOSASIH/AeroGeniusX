import numpy as np

def calculate_temperature_distribution(material_properties, heat_transfer, external_heat_flux):
    """
    Calculates the temperature distribution in an aircraft's thermal protection system.

    Args:
        material_properties (dict): Dictionary containing the thermal properties of the materials used in the system.
        heat_transfer (dict): Dictionary containing the heat transfer mechanisms and their corresponding coefficients.
        external_heat_flux (float): External heat flux applied to the system.

    Returns:
        str: Markdown code presenting a comprehensive analysis of the temperature distribution, heat flux,
             and thermal protection effectiveness in the aircraft's structure.
    """
    # Constants
    num_nodes = 100  # Number of nodes to discretize the structure
    length = 10.0  # Length of the structure in meters
    delta_x = length / num_nodes  # Distance between nodes

    # Initialize temperature array
    temperature = np.zeros(num_nodes)

    # Iterate over each node
    for i in range(1, num_nodes - 1):
        # Calculate heat flux at the node
        heat_flux = calculate_heat_flux(temperature[i-1], temperature[i+1], delta_x, heat_transfer)

        # Calculate temperature at the node
        temperature[i] = calculate_temperature(temperature[i], heat_flux, material_properties)

    # Calculate average temperature
    average_temperature = np.mean(temperature)

    # Calculate thermal protection effectiveness
    thermal_protection_effectiveness = calculate_thermal_protection_effectiveness(average_temperature, external_heat_flux)

    # Generate markdown output
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
        output += f"| {i+1} | {calculate_heat_flux(temperature[i-1], temperature[i+1], delta_x, heat_transfer):.2f} |\n"
    output += f"\n"
    output += f"### Thermal Protection Effectiveness\n\n"
    output += f"The average temperature in the structure is {average_temperature:.2f} °C.\n"
    output += f"The thermal protection effectiveness is {thermal_protection_effectiveness:.2f}%."

    return output

def calculate_heat_flux(temperature_prev, temperature_next, delta_x, heat_transfer):
    """
    Calculates the heat flux at a node based on the temperatures of the previous and next nodes.

    Args:
        temperature_prev (float): Temperature of the previous node.
        temperature_next (float): Temperature of the next node.
        delta_x (float): Distance between nodes.
        heat_transfer (dict): Dictionary containing the heat transfer mechanisms and their corresponding coefficients.

    Returns:
        float: Heat flux at the node.
    """
    # Calculate heat transfer coefficients
    conduction_coefficient = heat_transfer['conduction']
    convection_coefficient = heat_transfer['convection']

    # Calculate heat flux
    heat_flux = (conduction_coefficient * (temperature_prev - temperature_next) / delta_x) + convection_coefficient

    return heat_flux

def calculate_temperature(temperature, heat_flux, material_properties):
    """
    Calculates the temperature at a node based on the heat flux and material properties.

    Args:
        temperature (float): Current temperature at the node.
        heat_flux (float): Heat flux at the node.
        material_properties (dict): Dictionary containing the thermal properties of the material.

    Returns:
        float: Updated temperature at the node.
    """
    # Calculate thermal conductivity
    thermal_conductivity = material_properties['thermal_conductivity']

    # Calculate temperature change
    temperature_change = heat_flux / thermal_conductivity

    # Update temperature
    temperature += temperature_change

    return temperature

def calculate_thermal_protection_effectiveness(average_temperature, external_heat_flux):
    """
    Calculates the thermal protection effectiveness based on the average temperature and external heat flux.

    Args:
        average_temperature (float): Average temperature in the structure.
        external_heat_flux (float): External heat flux applied to the structure.

    Returns:
        float: Thermal protection effectiveness in percentage.
    """
    thermal_protection_effectiveness = (1 - (average_temperature / external_heat_flux)) * 100

    return thermal_protection_effectiveness

# Example usage
material_properties = {
    'thermal_conductivity': 0.5  # Thermal conductivity of the material in W/(m*K)
}

heat_transfer = {
    'conduction': 0.1,  # Conduction heat transfer coefficient in W/(m^2*K)
    'convection': 5.0  # Convection heat transfer coefficient in W/(m^2*K)
}

external_heat_flux = 100.0  # External heat flux in W/m^2

output = calculate_temperature_distribution(material_properties, heat_transfer, external_heat_flux)
print(output)
