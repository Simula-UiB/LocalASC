import stim

def add_circuit_level_noise(circuit: stim.Circuit, p1q: float = 0.001, p2q: float = 0.001, p_meas: float = 0.001) -> stim.Circuit:
    """
    Applies Circuit-Level noise to a well annotated circuit (proper TICK placement needed for correct application of idle noise).
    circuit is a stim circuit, p1q, p2q denote the error probability for one and two qubit gates and p_meas denotes error probability of a measurement  
    """

    #Total number of qubits to handle idle locations
    _total_qubits = circuit.num_qubits

    noisy_circuit = stim.Circuit()

    error_gates = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}

    single_qubit_gates = {"X", "Y", "Z", "H"}
    two_qubit_gates = {"CNOT", "SWAP", "CZ", "CX"}

    #Idle qubit handling
    all_qubits = set(range(_total_qubits))
    active_qubits_in_current_cycle = set()

    for instruction in circuit:

        #REPEAT blocks  
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # Unpack the tuple from the recursive call
            noisy_block = add_circuit_level_noise(
                circuit=instruction.body_copy(),
                p1q=p1q, 
                p2q=p2q, 
                p_meas=p_meas
            )
            noisy_circuit += noisy_block * instruction.repeat_count
            continue

        gate_name = instruction.name
        targets = instruction.targets_copy()

        if gate_name == "TICK":
            idle_qubits = list(all_qubits - active_qubits_in_current_cycle)
            if idle_qubits: 
                noisy_circuit.append("DEPOLARIZE1", idle_qubits, p1q)

            noisy_circuit.append(instruction)
            active_qubits_in_current_cycle.clear()
            continue

        #For every target, add them to active qubit in the current layer
        for t in targets:
            if t.is_qubit_target:
                active_qubits_in_current_cycle.add(t.value)

        #Final Data Measurements (M, MZ, MX)
        if gate_name in {"M", "MZ", "MX"}:
            error_type = "Z_ERROR" if gate_name == "MX" else "X_ERROR"
            noisy_circuit.append(error_type, targets, p_meas)
            noisy_circuit.append(gate_name, targets)

        #Syndrome Extraction Measurements & Resets (MR, MRX), since these are measurement and reinitialization, we append both measurement and initialization error
        elif gate_name in {"MR"}:
            noisy_circuit.append("X_ERROR", targets, p_meas)
            noisy_circuit.append(gate_name, targets)
            noisy_circuit.append("X_ERROR", targets, p1q)

        elif gate_name in {"MRX"}: 
            noisy_circuit.append("Z_ERROR", targets, p_meas)
            noisy_circuit.append(gate_name, targets)
            noisy_circuit.append("Z_ERROR", targets, p1q)

        #Initialization
        elif gate_name in {"R", "RZ"}:
            noisy_circuit.append(instruction)
            noisy_circuit.append("X_ERROR", targets, p1q) 

        elif gate_name in {"RX"}:
            noisy_circuit.append(instruction)
            noisy_circuit.append("Z_ERROR", targets, p1q) 

        #Standard Gates
        elif gate_name in two_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE2", targets, p2q)

        elif gate_name in single_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", targets, p1q)
        
        #Ignore error gates to not add duplicate noise
        elif gate_name in error_gates:
            continue
        #Append OBSERVABLES and DETECTORS
        else:
            noisy_circuit.append(instruction)

    return noisy_circuit

