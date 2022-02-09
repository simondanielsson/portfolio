package project1.Software.Instruction;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

public class Print implements Instruction {

    private Operand operand;
    
    public Print  (Operand operand){
        this.operand = operand;
    }

    public void execute(Memory memory, ProgramCounter counter) {
        // print the word stored in operand
        System.out.println(operand.getWord(memory));

        // Go to next instruction in the list
        counter.step();
    } 

    @Override
    public String describe() {
        return "Print " + operand.toString(); 
    }
}
