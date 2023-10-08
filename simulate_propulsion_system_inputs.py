engine_characteristics = {
    'max_thrust': 25000,  # Maximum thrust in Newtons
    'specific_fuel_consumption': 0.6,  # Specific fuel consumption in kg/s/N
    'efficiency': 0.85  # Efficiency of the propulsion system
}

airspeed = 300  # Airspeed in m/s
altitude = 10000  # Altitude in meters

output = simulate_propulsion_system(engine_characteristics, airspeed, altitude)
print(output)
