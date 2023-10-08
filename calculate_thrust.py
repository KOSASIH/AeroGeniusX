import math

def calculate_thrust(engine_power, airspeed):
    thrust = engine_power * (airspeed / 100)
    return thrust

def calculate_fuel_consumption(thrust, engine_efficiency):
    fuel_consumption = thrust / engine_efficiency
    return fuel_consumption

def calculate_efficiency(thrust, fuel_consumption):
    efficiency = thrust / fuel_consumption
    return efficiency

def simulate_propulsion_system(engine_power, engine_efficiency, airspeed, altitude):
    thrust = calculate_thrust(engine_power, airspeed)
    fuel_consumption = calculate_fuel_consumption(thrust, engine_efficiency)
    efficiency = calculate_efficiency(thrust, fuel_consumption)
    
    markdown_output = f"""
    # Propulsion System Simulation Results
    
    ## Engine Characteristics
    - Engine Power: {engine_power} kW
    - Engine Efficiency: {engine_efficiency} N/kW
    
    ## Flight Conditions
    - Airspeed: {airspeed} m/s
    - Altitude: {altitude} m
    
    ## Results
    - Thrust: {thrust} N
    - Fuel Consumption: {fuel_consumption} kg/s
    - Efficiency: {efficiency} N/kg
    """
    
    return markdown_output

# Example usage
engine_power = 5000  # kW
engine_efficiency = 0.8  # N/kW
airspeed = 250  # m/s
altitude = 10000  # m

markdown_output = simulate_propulsion_system(engine_power, engine_efficiency, airspeed, altitude)
print(markdown_output)
