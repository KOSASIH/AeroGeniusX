import numpy as np

def analyze_aircraft_stability(aircraft_dynamics, control_system_params, flight_conditions):
    # Extract inputs
    mass = aircraft_dynamics['mass']
    inertia = aircraft_dynamics['inertia']
    control_surface_areas = control_system_params['control_surface_areas']
    control_surface_deflections = control_system_params['control_surface_deflections']
    airspeed = flight_conditions['airspeed']
    altitude = flight_conditions['altitude']
    
    # Compute stability derivatives
    lift_derivative = compute_lift_derivative(airspeed, altitude)
    drag_derivative = compute_drag_derivative(airspeed, altitude)
    pitching_moment_derivative = compute_pitching_moment_derivative(airspeed, altitude)
    
    # Compute control effectiveness
    control_effectiveness = compute_control_effectiveness(control_surface_areas, control_surface_deflections)
    
    # Compute handling qualities
    handling_qualities = compute_handling_qualities(lift_derivative, drag_derivative, pitching_moment_derivative, control_effectiveness)
    
    # Prepare markdown output
    markdown_output = ""
    markdown_output += "## Stability and Control Analysis\n\n"
    markdown_output += "### Stability Derivatives\n\n"
    markdown_output += "- Lift Derivative: {}\n".format(lift_derivative)
    markdown_output += "- Drag Derivative: {}\n".format(drag_derivative)
    markdown_output += "- Pitching Moment Derivative: {}\n\n".format(pitching_moment_derivative)
    markdown_output += "### Control Effectiveness\n\n"
    markdown_output += "- Aileron Effectiveness: {}\n".format(control_effectiveness['aileron'])
    markdown_output += "- Elevator Effectiveness: {}\n".format(control_effectiveness['elevator'])
    markdown_output += "- Rudder Effectiveness: {}\n\n".format(control_effectiveness['rudder'])
    markdown_output += "### Handling Qualities\n\n"
    markdown_output += "- Handling Qualities Metric: {}\n".format(handling_qualities['metric'])
    markdown_output += "- Handling Qualities Rating: {}\n".format(handling_qualities['rating'])
    
    return markdown_output

def compute_lift_derivative(airspeed, altitude):
    # Perform computations
    lift_derivative = 0.5 * (airspeed ** 2) * np.cos(altitude)
    return lift_derivative

def compute_drag_derivative(airspeed, altitude):
    # Perform computations
    drag_derivative = 0.2 * (airspeed ** 2) * np.sin(altitude)
    return drag_derivative

def compute_pitching_moment_derivative(airspeed, altitude):
    # Perform computations
    pitching_moment_derivative = 0.1 * (airspeed ** 2) * np.sin(altitude)
    return pitching_moment_derivative

def compute_control_effectiveness(control_surface_areas, control_surface_deflections):
    # Perform computations
    aileron_effectiveness = 0.8 * control_surface_areas['aileron'] * control_surface_deflections['aileron']
    elevator_effectiveness = 0.6 * control_surface_areas['elevator'] * control_surface_deflections['elevator']
    rudder_effectiveness = 0.4 * control_surface_areas['rudder'] * control_surface_deflections['rudder']
    
    control_effectiveness = {
        'aileron': aileron_effectiveness,
        'elevator': elevator_effectiveness,
        'rudder': rudder_effectiveness
    }
    
    return control_effectiveness

def compute_handling_qualities(lift_derivative, drag_derivative, pitching_moment_derivative, control_effectiveness):
    # Perform computations
    handling_qualities_metric = lift_derivative / (drag_derivative + pitching_moment_derivative)
    
    if handling_qualities_metric > 0.8:
        handling_qualities_rating = "Excellent"
    elif handling_qualities_metric > 0.6:
        handling_qualities_rating = "Good"
    elif handling_qualities_metric > 0.4:
        handling_qualities_rating = "Fair"
    else:
        handling_qualities_rating = "Poor"
    
    handling_qualities = {
        'metric': handling_qualities_metric,
        'rating': handling_qualities_rating
    }
    
    return handling_qualities

# Example usage
aircraft_dynamics = {
    'mass': 1000,  # kg
    'inertia': 5000  # kg*m^2
}

control_system_params = {
    'control_surface_areas': {
        'aileron': 2,  # m^2
        'elevator': 1.5,  # m^2
        'rudder': 1  # m^2
    },
    'control_surface_deflections': {
        'aileron': 10,  # degrees
        'elevator': 5,  # degrees
        'rudder': 3  # degrees
    }
}

flight_conditions = {
    'airspeed': 100,  # m/s
    'altitude': 5000  # radians
}

output = analyze_aircraft_stability(aircraft_dynamics, control_system_params, flight_conditions)
print(output)