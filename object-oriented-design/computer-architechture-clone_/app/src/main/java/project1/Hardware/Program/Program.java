package project1.Hardware.Program;

import java.util.ArrayList;
import java.util.List;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;



public abstract class Program {
    
    protected List<Instruction> instructions = new ArrayList<>(); 

    protected final void add(Instruction instruction) {
        instructions.add(instruction); 
    }

    public Instruction getInstruction(ProgramCounter c) {
        return instructions.get(c.getIndex()); 
    }

    @Override
    public String toString() {
        var sb = new StringBuilder();

        for (int i = 0; i < instructions.size(); i++) {
            sb.append(i);
            sb.append(": ");
            sb.append(instructions.get(i).describe());
            sb.append("\n");
        }
        
        return sb.toString();
    }
}
