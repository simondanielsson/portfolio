package project1.Hardware.Instruction;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

public interface Instruction {

    /**
     * Execute this instruction
     * @param memory: Memory object
     * @param counter: counter keeping track of what instruction to be run next
     */
    public void execute(Memory memory, ProgramCounter counter); 

    /**
     * Returns a desription of the instruction.
     * @return description of the instruction 
     */
    public String describe();
}


