import math

def calculate_aerodynamic_forces(airspeed, altitude, aircraft_configuration):
    # Constants
    rho = 1.225  # Air density at sea level in kg/m^3
    g = 9.81  # Acceleration due to gravity in m/s^2
    
    # Aircraft configuration parameters
    wing_area = aircraft_configuration['wing_area']  # Wing area in square meters
    wing_span = aircraft_configuration['wing_span']  # Wing span in meters
    aspect_ratio = aircraft_configuration['aspect_ratio']  # Wing aspect ratio
    lift_coefficient = aircraft_configuration['lift_coefficient']  # Lift coefficient
    drag_coefficient = aircraft_configuration['drag_coefficient']  # Drag coefficient
    moment_coefficient = aircraft_configuration['moment_coefficient']  # Moment coefficient
    
    # Calculations
    dynamic_pressure = 0.5 * rho * airspeed**2  # Dynamic pressure in Pa
    lift_force = dynamic_pressure * wing_area * lift_coefficient  # Lift force in N
    drag_force = dynamic_pressure * wing_area * drag_coefficient  # Drag force in N
    pitching_moment = dynamic_pressure * wing_area * wing_span * moment_coefficient  # Pitching moment in Nm
    
    # Output markdown
    output = f"### Aerodynamic Analysis\n\n"
    output += f"**Airspeed:** {airspeed} m/s\n"
    output += f"**Altitude:** {altitude} m\n"
    output += f"**Wing Area:** {wing_area} m^2\n"
    output += f"**Wing Span:** {wing_span} m\n"
    output += f"**Aspect Ratio:** {aspect_ratio}\n\n"
    output += f"**Lift Force:** {lift_force} N\n"
    output += f"**Drag Force:** {drag_force} N\n"
    output += f"**Pitching Moment:** {pitching_moment} Nm\n"
    
    return output

# Example usage
airspeed = 100  # m/s
altitude = 5000  # m
aircraft_configuration = {
    'wing_area': 50,  # m^2
    'wing_span': 20,  # m
    'aspect_ratio': 8,
    'lift_coefficient': 1.2,
    'drag_coefficient': 0.04,
    'moment_coefficient': 0.02
}

aerodynamic_analysis = calculate_aerodynamic_forces(airspeed, altitude, aircraft_configuration)
print(aerodynamic_analysis)
