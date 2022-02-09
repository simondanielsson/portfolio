package project1.Software.Instruction;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;


public class JumpEq extends Jump {

    private Operand o1, o2;

    public JumpEq  (int next, Operand o1, Operand o2) {
        super(next);
        this.o1 = o1;
        this.o2 = o2;
    }

    public void execute(Memory memory, ProgramCounter counter) {
        // check if the two walues are equal jump to index "next" according to jump 
        if (o1.getWord(memory).equals(o2.getWord(memory)) ){
            super.execute(memory, counter);
        } else {
            counter.step();
        }
    } 

    @Override
    public String describe() {
        return "Jump to " + String.valueOf(next) + " if " + o1.toString() + " == " + o2.toString();
    } 



    
}