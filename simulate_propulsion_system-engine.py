import math

def calculate_thrust(max_thrust: float, altitude: float) -> float:
    # Constants
    GRAVITY = 9.81  # m/s^2
    GAS_CONSTANT = 0.0065  # specific gas constant for air
    TEMPERATURE_AT_SEA_LEVEL = 288.15  # temperature at sea level in Kelvin

    # Calculate thrust
    pressure_ratio = (1 - (altitude * 0.001 * GAS_CONSTANT / TEMPERATURE_AT_SEA_LEVEL)) ** (GRAVITY / (GAS_CONSTANT * 287.1))
    thrust = max_thrust * pressure_ratio

    return thrust

def calculate_fuel_consumption(thrust: float, specific_fuel_consumption: float) -> float:
    fuel_consumption = thrust / specific_fuel_consumption
    return fuel_consumption

def calculate_efficiency(thrust: float, fuel_consumption: float, gravity: float, airspeed: float) -> float:
    efficiency = thrust / (fuel_consumption * gravity * airspeed)
    return efficiency

def generate_markdown_output(engine_characteristics: dict, airspeed: float, altitude: float, thrust: float, fuel_consumption: float, efficiency: float) -> str:
    markdown_output = f"""
    ## Propulsion System Analysis

    ### Inputs:
    - Engine Characteristics:
        - Thrust Specific Fuel Consumption: {engine_characteristics['thrust_specific_fuel_consumption']} kg/N/h
        - Maximum Thrust: {engine_characteristics['max_thrust']} N
    - Airspeed: {airspeed} m/s
    - Altitude: {altitude} m

    ### Results:
    - Thrust: {thrust} N
    - Fuel Consumption: {fuel_consumption} kg/h
    - Efficiency: {efficiency} N/(kg/s)

    """

    return markdown_output

def simulate_propulsion_system(engine_characteristics: dict, airspeed: float, altitude: float) -> str:
    """
    Simulate a propulsion system with given engine characteristics, airspeed, and altitude.

    Parameters:
    engine_characteristics (dict
