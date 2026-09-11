import stim
from typing import Tuple

def add_circuit_level_noise(
    circuit: stim.Circuit, 
    p1q: float = 0.001, 
    p2q: float = 0.001, 
    p_meas: float = 0.001, 
    _total_qubits=None,
    skip_final_measurements: bool = True,
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

    return noisy_circuit, idle_error_count