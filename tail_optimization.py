```python
import numpy as np
from scipy.optimize import minimize

def tail_optimization(tail_geometry, aircraft_dynamics, flight_conditions):
    def objective_function(x):
        # Extract tail shape parameters
        tail_length, tail_width, tail_height = x
        
        # Calculate stability and maneuverability characteristics
        stability = calculate_stability(tail_length, tail_width, tail_height, aircraft_dynamics, flight_conditions)
        maneuverability = calculate_maneuverability(tail_length, tail_width, tail_height, aircraft_dynamics, flight_conditions)
        
        # Define the objective function to be minimized
        objective = -stability + maneuverability
        
        return objective
    
    def calculate_stability(tail_length, tail_width, tail_height, aircraft_dynamics, flight_conditions):
        # Calculate stability based on tail geometry, aircraft dynamics, and flight conditions
        # ...
        stability = ...
        return stability
    
    def calculate_maneuverability(tail_length, tail_width, tail_height, aircraft_dynamics, flight_conditions):
        # Calculate maneuverability based on tail geometry, aircraft dynamics, and flight conditions
        # ...
        maneuverability = ...
        return maneuverability
    
    # Define the initial guess for tail shape parameters
    initial_guess = [tail_geometry["length"], tail_geometry["width"], tail_geometry["height"]]
    
    # Define the bounds for tail shape parameters
    bounds = [(0, None), (0, None), (0, None)]
    
    # Optimize the tail shape parameters
    result = minimize(objective_function, initial_guess, bounds=bounds)
    
    # Extract the optimized tail shape parameters
    optimized_tail_length, optimized_tail_width, optimized_tail_height = result.x
    
    # Calculate stability and maneuverability characteristics with the optimized tail shape parameters
    optimized_stability = calculate_stability(optimized_tail_length, optimized_tail_width, optimized_tail_height, aircraft_dynamics, flight_conditions)
    optimized_maneuverability = calculate_maneuverability(optimized_tail_length, optimized_tail_width, optimized_tail_height, aircraft_dynamics, flight_conditions)
    
    # Output markdown code for the optimized tail shape parameters and stability/maneuverability characteristics
    output = f"""
    # Optimized Tail Shape Parameters
    
    - Length: {optimized_tail_length}
    - Width: {optimized_tail_width}
    - Height: {optimized_tail_height}
    
    # Stability Characteristics
    
    - Stability: {optimized_stability}
    
    # Maneuverability Characteristics
    
    - Maneuverability: {optimized_maneuverability}
    """
    
    return output

# Example usage
tail_geometry = {
    "length": 2.5,
    "width": 1.2,
    "height": 0.4
}

aircraft_dynamics = {
    # Define aircraft dynamics parameters
    # ...
}

flight_conditions = {
    # Define flight conditions
    # ...
}

output = tail_optimization(tail_geometry, aircraft_dynamics, flight_conditions)
print(output)
```

The code above provides a framework for optimizing the shape of an aircraft tail for improved stability and maneuverability. It defines an objective function that takes tail shape parameters as inputs and calculates the stability and maneuverability characteristics based on the given aircraft dynamics and flight conditions. The code then uses the `scipy.optimize.minimize` function to find the optimal tail shape parameters that minimize the objective function. Finally, it outputs the optimized tail shape parameters and the corresponding stability and maneuverability characteristics in markdown format.
