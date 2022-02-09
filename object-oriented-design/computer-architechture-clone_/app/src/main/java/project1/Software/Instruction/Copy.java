package project1.Software.Instruction;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;
import project1.Hardware.Operand.Adress.Adress;
import project1.Hardware.Operand.Word.Word;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

public class Copy implements Instruction {

    private Operand operand;
    private Adress targetAdress;

    public Copy (Operand operand, Adress adress){
        this.operand = operand;
        this.targetAdress = adress;
    }

    public void execute(Memory memory, ProgramCounter counter) {
        // Find the object corresponding to the target adress
        Word target = targetAdress.getWord(memory);         // Det här går säkert skriva snyggare...
        // copy the value from  operand other to the target adress
        target.copy(operand.getWord(memory));

        // Go to next instruction in the list
        counter.step();
    }     

    public String describe() {
        return "Copy " + operand.toString() + " to " + targetAdress.toString(); 
    }
}
