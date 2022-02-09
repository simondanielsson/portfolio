package project1.Software.Instruction;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Memory.Memory;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;


public class Halt implements Instruction {

    public Halt (){};

    public void execute(Memory memory, ProgramCounter counter) {
        // stop the process
        counter.halt(); 
    }

    @Override
    public String describe() {
        return "Halt";
    } 

    

}
