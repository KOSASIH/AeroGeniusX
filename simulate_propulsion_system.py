```python
import math

def simulate_propulsion_system(engine_characteristics, airspeed, altitude):
    # Extract engine characteristics
    max_thrust = engine_characteristics['max_thrust']
    specific_fuel_consumption = engine_characteristics['specific_fuel_consumption']
    efficiency = engine_characteristics['efficiency']
    
    # Calculate thrust
    thrust = calculate_thrust(max_thrust, airspeed, altitude)
    
    # Calculate fuel consumption
    fuel_consumption = calculate_fuel_consumption(thrust, specific_fuel_consumption)
    
    # Calculate efficiency
    propulsion_efficiency = calculate_efficiency(thrust, fuel_consumption)
    
    # Output markdown code
    output = f"## Propulsion System Analysis\n\n"
    output += f"**Thrust:** {thrust} N\n"
    output += f"**Fuel Consumption:** {fuel_consumption} kg/s\n"
    output += f"**Efficiency:** {propulsion_efficiency}\n"
    
    return output

def calculate_thrust(max_thrust, airspeed, altitude):
    # Calculate thrust using simplified model
    rho = calculate_air_density(altitude)
    thrust = max_thrust * (airspeed / 343) * math.sqrt(rho / 1.225)
    return thrust

def calculate_air_density(altitude):
    # Calculate air density using simplified model
    temperature = 288.15 - 0.0065 * altitude
    pressure = 101325 * (temperature / 288.15) ** (9.81 / (287.05 * 0.0065))
    rho = pressure / (287.05 * temperature)
    return rho

def calculate_fuel_consumption(thrust, specific_fuel_consumption):
    # Calculate fuel consumption based on thrust and specific fuel consumption
    fuel_consumption = thrust / specific_fuel_consumption
    return fuel_consumption

def calculate_efficiency(thrust, fuel_consumption):
    # Calculate efficiency based on thrust and fuel consumption
    efficiency = thrust / (fuel_consumption * 9.81)
    return efficiency
```

To use this code, you can call the `simulate_propulsion_system` function with the appropriate inputs:

```python
engine_characteristics = {
    'max_thrust': 25000,  # Maximum thrust in Newtons
    'specific_fuel_consumption': 0.6,  # Specific fuel consumption in kg/s/N
    'efficiency': 0.85  # Efficiency of the propulsion system
}

airspeed = 300  # Airspeed in m/s
altitude = 10000  # Altitude in meters

output = simulate_propulsion_system(engine_characteristics, airspeed, altitude)
print(output)
```

This code will output a markdown-formatted analysis of the thrust, fuel consumption, and efficiency of the propulsion system based on the given inputs.
