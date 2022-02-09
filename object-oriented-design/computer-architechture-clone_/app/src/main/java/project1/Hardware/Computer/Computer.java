package project1.Hardware.Computer;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Program.Program;
import project1.Hardware.Program.ProgramCounter.IntProgramCounter;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

public class Computer {

    private Memory memory;
    private Program program;

    public Computer  (Memory memory) {
        this.memory = memory;
    }

    public void load(Program program) {
        this.program = program;
    }
    
    public void run() {    
        ProgramCounter counter = new IntProgramCounter(); 

        while (counter.stillRunning()) {
            program.getInstruction(counter).execute(memory, counter); 
        }
    }

}
