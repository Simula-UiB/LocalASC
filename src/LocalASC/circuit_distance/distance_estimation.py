import stim
from codedistance import circuitDistance
from src.LocalASC.circuit_builder.noise import add_circuit_level_noise

def compute_circuit_distance(circuit : stim.Circuit, method : str, params : dict, seed : int =42):
    """
    Wrapper for the circuitDistance method with an intermediate Circuit-level noise application step.
    Given a circuit, performs the specified 'method' (e.g QDistRndMW, QDistEvol, UndetectableErrorStim, etc...) to compute or estimate circuit distance under circuit-level noise.
    params is the parameter dictionary from circuitDistance for the given method and seed for reproducibility.
    """
    #Apply noise
    noisy_circuit = add_circuit_level_noise(circuit)
    #Perform distance-finding method
    code_dict = circuitDistance(noisy_circuit, method, params, seed)

    return code_dict