import math

def calculate_air_density(altitude: float) -> float:
    """Calculate the air density at a given altitude in kg/m^3.

    Args:
        altitude (float): The altitude in meters.

    Returns:
        float: The air density in kg/m^3.
    """
    rho = 1.225 * (1 - 0.0065 * altitude / 288.15) ** 5.257
    return rho

def calculate_aerodynamic_forces(aircraft_configuration: dict, airspeed: float, altitude: float) -> str:
    """Calculate aerodynamic forces and moments for an aircraft.

    Args:
        aircraft_configuration (dict): The aircraft configuration dictionary.
        airspeed (float): The airspeed in m/s.
        altitude (float): The altitude in meters.

    Returns:
        str: A formatted string with the aerodynamic forces and moments.
    """
    # Aircraft configuration parameters
    wing_area = aircraft_configuration['wing_area']  # Wing area in square meters
    wing_span = aircraft_configuration['wing_span']  # Wing span in meters
    aspect_ratio = aircraft_configuration['aspect_ratio']  # Wing aspect ratio
    lift_coefficient = aircraft_configuration['lift_coefficient']  # Lift coefficient
    drag_coefficient = aircraft_configuration['drag_coefficient']  # Drag coefficient
    moment_coefficient = aircraft_configuration['moment_coefficient']  # Moment coefficient

    # Calculations
    rho = calculate_air_density(altitude)
    dynamic_pressure = 0.5 * rho * airspeed**2  # Dynamic pressure in Pa
    lift_force = dynamic_pressure * wing_area * lift_coefficient  # Lift force in N
    drag_force = dynamic_pressure * wing_area * drag_coefficient  # Drag force in N
    pitching_moment = dynamic_pressure * wing_area * wing_span * moment_coefficient  # Pitching moment in Nm

    # Output markdown
    output = f"## Aerodynamic Analysis\n\n"
    output += f"**Airspeed:** {airspeed} m/s\n"
    output += f"**Altitude:** {altitude} m\n"
    output += f"**Wing Area:** {wing_area} m^2\n"
    output += f"**Wing Span:** {wing_span} m\n"
    output += f"**Aspect Ratio:** {aspect_ratio}\n\n"
    output += f"**Lift Force:** {lift_force} N\n"
    output += f"**Drag Force:** {drag_force} N\n"
    output += f"**Pitching Moment:** {pitching_moment} Nm\n
