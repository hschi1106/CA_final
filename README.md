# CA_final – Verilog CPU

This repository contains a course project for a Computer Architecture class:  
a simple CPU implemented in Verilog, along with a testbench, memory module, and several example programs.

The CPU is designed to execute a custom instruction set defined for the course (see `Instruction_set.pdf`) and is evaluated using the provided testbench and sample programs such as arithmetic, GCD, and sorting.

---

## Features

- **Custom Instruction Set**
  - Instruction formats and opcodes defined in `Instruction_set.pdf`
  - Basic arithmetic and logical operations
  - Memory access instructions
  - Branch / control-flow support

- **CPU Core (`CPU.v`)**
  - Instruction fetch, decode, execute, memory, and write-back stages (course-level implementation)
  - ALU operations for arithmetic and logical instructions
  - Register file and program counter handling
  - Simple control logic to drive datapath components

- **Memory Module (`memory.v`)**
  - Instruction and data memory for the CPU
  - Preloaded with assembled programs for testing
  - Simple interface used by `CPU.v` and `testbench.v`

- **Testbench (`testbench.v`)**
  - Instantiates the CPU and memory
  - Provides clock and reset signals
  - Monitors execution and prints results for verification

- **Example Programs**
  - `arithmetic/` – basic arithmetic operations
  - `gcd/` – greatest common divisor computation
  - `sort/` – simple sorting algorithm implementation

Detailed design, diagrams, and explanations are available in `CA_final_project.pdf` / `report.pdf`.

---

## Repository Structure

```text
.
├── CPU.v                 # Top-level CPU module
├── memory.v              # Memory module
├── testbench.v           # Testbench for simulation
├── Instruction_set.pdf   # Instruction formats and opcode specification
├── CA_final_project.pdf  # Project slides / documentation (course deliverable)
├── report.pdf            # Written report (if provided)
├── arithmetic/           # Example program: arithmetic operations
├── gcd/                  # Example program: GCD
├── sort/                 # Example program: sorting
└── .gitignore
