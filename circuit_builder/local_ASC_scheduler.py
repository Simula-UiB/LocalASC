import numpy
import json, sys
import stim

def asc_circuit(Hx, Hz) -> stim.Circuit:
    circuit = stim.Circuit()
    return circuit 

def add_circuit_level_noise(circuit : stim.Circuit, p1q : float = 0.001, p2q : float = 0.001, p_meas : float = 0.001, _total_qubits = None) -> stim.Circuit:
    if _total_qubits is None:
        _total_qubits = circuit.num_qubits


    noisy_circuit = stim.Circuit()


    single_qubit_gates = {"X", "Y", "Z", "H"}
    two_qubit_gates = {"CNOT", "SWAP", "CZ", "CX"}

    all_qubits = set(range(_total_qubits))
    active_qubits_in_current_cycle = set()



    for instruction in circuit:
        

        if isinstance(instruction, stim.CircuitRepeatBlock):
            noisy_block = add_circuit_level_noise(
                circuit= instruction.body_copy(),
                p1q=p1q, 
                p2q=p2q, 
                p_meas=p_meas, 
                _total_qubits=_total_qubits
            )
            noisy_circuit += noisy_block * instruction.repeat_count
            continue

        gate_name = instruction.name
        targets = instruction.targets_copy()

        if gate_name == "TICK":
            idle_qubits = list(all_qubits - active_qubits_in_current_cycle)
            
            if idle_qubits:
                idle_qubits.sort() 
                noisy_circuit.append("DEPOLARIZE1", idle_qubits, p1q)
            
            noisy_circuit.append(instruction)
            active_qubits_in_current_cycle.clear()
            continue

        for t in targets:
            if t.is_qubit_target:
                active_qubits_in_current_cycle.add(t.value)

        if gate_name in {"M", "MX", "MY", "MZ", "MR"}:
            noisy_circuit.append(gate_name, targets, p_meas)
            
        elif gate_name in {"R", "RX", "RY", "RZ"}:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", targets, p_meas)

        elif gate_name in two_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE2", targets, p2q)

        elif gate_name in single_qubit_gates:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", targets, p1q)

        else:
            noisy_circuit.append(instruction)
            
    return noisy_circuit

def add_phenomenological_noise(circuit : stim.Circuit) -> stim.Circuit:
    return circuit