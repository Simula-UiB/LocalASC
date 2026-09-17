import stim
from typing import Tuple


def get_noisy_circuit(noiseless_circuit: stim.Circuit, p: float):
    """
    Applies gate noise, idle noise, and state preparation/measurement noise.
    
    NOTES:
    1. Measurements (M): Noise is applied BEFORE to flip the result.
       - Uses X_ERROR(p) for Z-basis M so the flip probability is exactly p.
    2. Resets/Gates: Noise is applied AFTER to scramble the resulting state.
    3. Idle Noise: Applied to any qubit not involved in an operation during a TICK.
    """
    if p == 0:
        return noiseless_circuit
        
    noisy_circuit = stim.Circuit()
    
    # 1. Define the Universe of Qubits
    all_qubits_set = set()
    for instruction in noiseless_circuit:
        for t in instruction.targets_copy(): #type: ignore
            if t.is_qubit_target:
                all_qubits_set.add(t.value)
                
    # 2. Inject Noise
    busy_qubits = set()
    first_tick = True
    
    for instruction in noiseless_circuit:
        
        # --- A. Handle Time Steps (Idle Noise) ---
        if instruction.name == "TICK":
            # Only skip if the circuit literally started with an empty TICK
            if first_tick and len(busy_qubits) == 0:
                first_tick = False
                noisy_circuit.append("TICK") #type: ignore
                continue
            
            first_tick = False # Make sure we turn the flag off

            # Apply idle noise to any qubit that was NOT busy in the previous moment
            idle_qubits = list(all_qubits_set - busy_qubits)
            if idle_qubits:
                noisy_circuit.append("DEPOLARIZE1", idle_qubits, p)
            
            busy_qubits = set()
            noisy_circuit.append("TICK") #type: ignore
            continue

        # --- B. Handle Instructions ---
        targets = instruction.targets_copy() #type: ignore
        
        # Track busy qubits for the current moment
        for t in targets:
            if t.is_qubit_target:
                busy_qubits.add(t.value)
        
        # CASE 1: Measurements (Noise BEFORE)
        # We must flip the qubit *before* Stim records the measurement result.
        if instruction.name == "M":
            # X_ERROR flips Z-basis eigenstates (|0> <-> |1>)
            noisy_circuit.append("X_ERROR", targets, p)
            noisy_circuit.append(instruction)
            
        elif instruction.name == "MX":
            # Z_ERROR flips X-basis eigenstates (|+> <-> |->)
            noisy_circuit.append("Z_ERROR", targets, p)
            noisy_circuit.append(instruction)
            
        # CASE 2: Resets & Gates (Noise AFTER)
        else:
            noisy_circuit.append(instruction)
            
            # Apply noise immediately after the operation finishes
            if instruction.name in ["R", "RZ"]:
                 # Reset to |0>. Noise is X (bit flip).
                 noisy_circuit.append("X_ERROR", targets, p)

            elif instruction.name == "RX":
                 # Reset to |+>. Noise is Z (phase flip).
                 noisy_circuit.append("Z_ERROR", targets, p)
                
            elif instruction.name in ["H", "S", "X", "Y", "Z", "SQRT_X", "SQRT_Y", "I"]:
                # Single-qubit gate error
                noisy_circuit.append("DEPOLARIZE1", targets, p)
                
            elif instruction.name in ["CNOT", "CX", "CY", "CZ", "SWAP"]:
                # Two-qubit gate error
                noisy_circuit.append("DEPOLARIZE2", targets, p)

# --- C. Handle the Final Round Idle ---
    # Only apply if there are leftover operations that weren't closed by a TICK
    if not first_tick and len(busy_qubits) > 0: 
        idle_qubits = list(all_qubits_set - busy_qubits)
        if idle_qubits:
            noisy_circuit.append("DEPOLARIZE1", idle_qubits, p)

    dem = noisy_circuit.detector_error_model()
    shape = dem.num_detectors, dem.num_errors
    print(f"Noisy circuit has shape {shape}")
    
    return noisy_circuit, 0

def add_circuit_level_noise(
    circuit: stim.Circuit, 
    p1q: float = 0.001, 
    p2q: float = 0.001, 
    p_meas: float = 0.001, 
    _total_qubits=None,
    skip_final_measurements: bool = False,
    apply_idle_noise_per_tick: bool = True,
    _is_top_level: bool = True
) -> Tuple[stim.Circuit, int]:

    if _total_qubits is None:
        _total_qubits = circuit.num_qubits

    noisy_circuit = stim.Circuit()
    idle_error_count = 0  # Initialize the counter

    single_qubit_gates = {"X", "Y", "Z", "H"}
    two_qubit_gates = {"CNOT", "SWAP", "CZ", "CX"}

    # Instructions that are allowed to appear after a "final" measurement
    allowed_final_ops = {
        "M", "MX", "MZ", "MPP", 
        "DETECTOR", "OBSERVABLE_INCLUDE", 
        "TICK", "SHIFT_COORDS", "QUBIT_COORDS", "POLYGON"
    }

    all_qubits = set(range(_total_qubits))
    active_qubits_in_current_cycle = set()

    for i, instruction in enumerate(circuit): #type: ignore
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # Unpack the tuple from the recursive call
            noisy_block, block_idle_count = add_circuit_level_noise(
                circuit=instruction.body_copy(),
                p1q=p1q, 
                p2q=p2q, 
                p_meas=p_meas, 
                _total_qubits=_total_qubits,
                skip_final_measurements=skip_final_measurements,
                apply_idle_noise_per_tick=apply_idle_noise_per_tick,
                _is_top_level=False # Measurements inside loops are never final
            )
            noisy_circuit += noisy_block * instruction.repeat_count
            idle_error_count += block_idle_count * instruction.repeat_count # Multiply by loop iterations
            continue

        gate_name = instruction.name
        targets = instruction.targets_copy()

        if gate_name == "TICK":
            if apply_idle_noise_per_tick:
                idle_qubits = list(all_qubits - active_qubits_in_current_cycle)
                if idle_qubits:
                    idle_qubits.sort() 
                    noisy_circuit.append("DEPOLARIZE1", idle_qubits, p1q)
                    idle_error_count += len(idle_qubits) # Add to the counter

            noisy_circuit.append(instruction)
            active_qubits_in_current_cycle.clear()
            continue

        for t in targets:
            if t.is_qubit_target:
                active_qubits_in_current_cycle.add(t.value)

        # Final Data Measurements (M, MZ, MX)
        if gate_name in {"M", "MZ", "MX"}:
            apply_noise = True

            if skip_final_measurements and _is_top_level:
                # Look ahead to see if any gates or resets occur after this measurement
                is_final = True
                for j in range(i + 1, len(circuit)):
                    future_inst = circuit[j]
                    if isinstance(future_inst, stim.CircuitRepeatBlock) or future_inst.name not in allowed_final_ops:
                        is_final = False
                        break

                if is_final:
                    apply_noise = False

            if apply_noise:
                error_type = "Z_ERROR" if gate_name == "MX" else "X_ERROR"
                noisy_circuit.append(error_type, targets, p_meas)

            noisy_circuit.append(gate_name, targets)

        # Syndrome Extraction Measurements & Resets (MR, MRX)
        elif gate_name in {"MR"}:
            noisy_circuit.append("X_ERROR", targets, p_meas)
            noisy_circuit.append(gate_name, targets)
            noisy_circuit.append("X_ERROR", targets, p1q)

        elif gate_name in {"MRX"}: 
            noisy_circuit.append("Z_ERROR", targets, p_meas)
            noisy_circuit.append(gate_name, targets)
            noisy_circuit.append("Z_ERROR", targets, p1q)

        # Standard Initialization
        elif gate_name in {"R", "RZ"}:
            noisy_circuit.append(instruction)
            noisy_circuit.append("X_ERROR", targets, p1q) 

        elif gate_name in {"RX"}:
            noisy_circuit.append(instruction)
            noisy_circuit.append("Z_ERROR", targets, p1q) 

        # Standard Gates
        elif gate_name in two_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE2", targets, p2q)

        elif gate_name in single_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", targets, p1q)

        else:
            noisy_circuit.append(instruction)

    dem = noisy_circuit.detector_error_model()
    shape = dem.num_detectors, dem.num_errors
    print(f"Noisy cxircuit has shape {shape}")

    return noisy_circuit, idle_error_count