import math

def calculate_thrust(engine_characteristics, airspeed, altitude):
    # Calculate thrust based on engine characteristics
    thrust = engine_characteristics['thrust_constant'] * engine_characteristics['thrust_coefficient'] * airspeed ** 2
    
    # Adjust thrust for altitude
    thrust *= (1 - (altitude * engine_characteristics['altitude_loss']))
    
    return thrust

def calculate_fuel_consumption(engine_characteristics, thrust):
    # Calculate fuel consumption based on thrust
    fuel_consumption = thrust / engine_characteristics['specific_fuel_consumption']
    
    return fuel_consumption

def calculate_efficiency(thrust, fuel_consumption):
    # Calculate efficiency as thrust divided by fuel consumption
    efficiency = thrust / fuel_consumption
    
    return efficiency

# Example input values
engine_characteristics = {
    'thrust_constant': 0.5,
    'thrust_coefficient': 0.8,
    'altitude_loss': 0.001,
    'specific_fuel_consumption': 0.02
}
airspeed = 100
altitude = 5000

# Calculate thrust, fuel consumption, and efficiency
thrust = calculate_thrust(engine_characteristics, airspeed, altitude)
fuel_consumption = calculate_fuel_consumption(engine_characteristics, thrust)
efficiency = calculate_efficiency(thrust, fuel_consumption)

# Output markdown code for analysis
output = f"""
## Propulsion System Analysis

### Engine Characteristics
- Thrust Constant: {engine_characteristics['thrust_constant']}
- Thrust Coefficient: {engine_characteristics['thrust_coefficient']}
- Altitude Loss: {engine_characteristics['altitude_loss']}
- Specific Fuel Consumption: {engine_characteristics['specific_fuel_consumption']}

### Input Values
- Airspeed: {airspeed}
- Altitude: {altitude}

### Results
- Thrust: {thrust}
- Fuel Consumption: {fuel_consumption}
- Efficiency: {efficiency}
"""

print(output)
