package project1.Software.Instruction;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Memory.Memory;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

public class Jump implements Instruction {

    public int next;

    public Jump  (int next){
        this.next  = next;
    }

    public void execute(Memory memory, ProgramCounter counter) {
        // Jump to instruction with index "next" in the list
        counter.jumpTo(next);
    }

    @Override
    public String describe() {
        return "Jump to " + String.valueOf(next);
    } 

    
}
